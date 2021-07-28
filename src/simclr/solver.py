import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torchvision.models.densenet import _load_state_dict
from tqdm import tqdm
from torchvision.models import resnet18, resnet50
from model import ResNetSimCLR, AttGANGenerator
from torch.utils.data import DataLoader
from utils import AverageMeter
import dataset
import os


class Solver():
    def __init__(self, args):
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.n_epochs = args.n_epochs
        self.ckpt_save_epoch = args.ckpt_save_epoch
        self.ckpt_dir = args.ckpt_dir
        self.log_dir = args.log_dir
        self.temperature = args.temperature
        self.s_epoch = 1

        # Data
        self.dset_dir = args.dset_dir
        self.batch_size = args.batch_size

        # Data augmentation
        self.attgan_transform = args.attgan_transform
        self.random_resized_crop = args.random_resized_crop
        self.img_sz = args.img_sz

        if args.attgan_transform:
            self.attgan = AttGANGenerator().to(self.device)
            state_dict = torch.load(args.attgan_dir)
            self.attgan.load_state_dict(state_dict)
            self.attgan.eval()

        # Train data
        if args.dataset == "LFW":
            self.train_data = dataset.PeopleLFWPair(
                root=self.dset_dir, split='train',
                transform=dataset.get_simclr_pipeline_transform(args))
        elif args.dataset == "CelebA":
            self.train_data = dataset.CelebAPair(
                root=self.dset_dir, split='train',
                transform=dataset.get_simclr_pipeline_transform(args))
        elif args.dataset == "CASIA":
            self.train_data = dataset.CASIAPair(
                root=self.dset_dir, split='train',
                transform=dataset.get_simclr_pipeline_transform(args))
        else:
            raise NotImplementedError("Dataset is not implemented")

        self.train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=False, drop_last=True)

        # Test data
        self.test_data = dataset.PairsLFW(
            root=self.dset_dir, split='test',
            transform=transforms.Compose([
                # transforms.Resize(args.img_sz),
                transforms.ToTensor(),
            ]))

        self.test_loader = DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=False)

        # Model
        if args.base_encoder == 'resnet18':
            base_encoder = resnet18
        elif args.base_encoder == 'resnet50':
            base_encoder = resnet50
        else:
            raise NotImplementedError(
                "base encoder must be resnet18 or resnet50")

        self.model = ResNetSimCLR(
            base_encoder=base_encoder,
            feature_dim=args.feature_dim,
            projection_dim=args.projection_dim).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.lr, weight_decay=args.wd)

        # Checkpoint load
        if args.ckpt_load:
            ckpt = torch.load(args.ckpt_load, map_location=self.device)
            self.model.load_state_dict(ckpt["model_states"])
            self.optimizer.load_state_dict(ckpt["optim_states"])
            self.s_epoch = ckpt["epoch"]

    def info_nce_loss(self, features):
        # label: nviews x batchsize,
        labels = torch.cat([torch.arange(self.batch_size)
                           for i in range(2)], dim=0)

        # labels: (nviews x batch) x (nviews x batch),
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        # cosine similarity matrix: (nviews x batch) x (nviews x batch)
        similarity_matrix = torch.matmul(features, features.T)

        # mask: (nviews x batch) x (nviews x batch)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)

        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1)

        # positives: (nviews x batch) x 1
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # negatives: (nviews x batch) x ((nviews x batch) - 2)
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1)

        # logits: (nviews x batch) x ((nviews x batch) - 1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return F.cross_entropy(logits, labels)

    def train(self):
        writer = SummaryWriter(self.log_dir)
        scaler = GradScaler()

        for epoch in range(self.s_epoch, self.s_epoch + self.n_epochs + 1):
            train_loss = AverageMeter()
            self.model.train()
            train_bar = tqdm(self.train_loader)
            for pos_1, pos_2, _ in train_bar:
                bs = pos_1.size(0)
                pos_1, pos_2 = pos_1.to(self.device), pos_2.to(self.device)
                
                with torch.no_grad():
                    if self.attgan_transform:
                        att_b_1 = torch.rand((bs, 13)) * 2 - 1
                        att_b_2 = torch.rand((bs, 13)) * 2 - 1
                        att_b_1 = att_b_1.to(self.device)
                        att_b_2 = att_b_2.to(self.device)
                        pos_1 = self.attgan(pos_1, att_b_1)
                        pos_2 = self.attgan(pos_2, att_b_2)

                self.optimizer.zero_grad()
                with autocast():
                    _, out_1 = self.model(pos_1)
                    _, out_2 = self.model(pos_2)
                    # [2*B, D]
                    out = torch.cat([out_1, out_2], dim=0)
                    loss = self.info_nce_loss(out)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                train_bar.set_description(
                    "Epoch: {} | Train loss: {:.4f}".format(epoch, train_loss.avg))
                train_loss.update(loss.item(), bs)

            if epoch % self.ckpt_save_epoch == 0:
                self.save_checkpoint(epoch, 'epoch_{}.pth'.format(epoch))
            self.save_checkpoint(epoch)
            auc = self.evaluate()
            train_bar.write("AUC: {:.4f}".format(auc))

            writer.add_scalar('auc', auc, epoch)

        self.save_checkpoint(self.n_epochs)
        writer.close()

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate on verification task
        """
        # evaluation mode
        self.model.eval()

        # batch / encode / decode
        all_preds_f = []
        all_targets = []
        for img1, img2, targets in self.test_loader:
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            targets = targets.to(self.device)
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1
            f1, _ = self.model(img1)
            f2, _ = self.model(img2)

            preds = F.cosine_similarity(f1, f2)
            all_preds_f.append(preds)
            all_targets.append(targets)

        all_preds = torch.cat(all_preds_f, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        self.auc = roc_auc_score(
            all_targets.long().cpu().numpy(), all_preds.cpu().numpy())
        return self.auc

    @torch.no_grad()
    def show_img(self):
        pos_1 = self.train_data[0]

    def save_checkpoint(self, epoch, ckptname='last.pth'):
        states = {'epoch': epoch,
                  'model_states': self.model.state_dict(),
                  'optim_states': self.optimizer.state_dict()}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))
        torch.save(states, filepath)
