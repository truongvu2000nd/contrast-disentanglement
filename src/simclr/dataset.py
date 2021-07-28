from warnings import simplefilter
from PIL import Image
from torch.utils import data
from torchvision import transforms
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, CelebA
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import Resize
import PIL

from model import AttGANGenerator


class PairsLFW(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        if split == "train":
            self.txt_file = "pairsDevTrain.txt"
        elif split == "test":
            self.txt_file = "pairsDevTest.txt"
        else:
            raise ValueError(
                "Split \"{}\" is not in (\"train\", \"test\").".format(split))

        self.root = root
        self.num_matched, self.num_mismatched, self.pairs = self._read_txt()
        self.targets = torch.cat(
            (torch.ones(self.num_matched), torch.zeros(self.num_mismatched)))
        self.transform = transform
        self.target_transform = target_transform

    def _read_txt(self):
        with open(os.path.join(self.root, self.txt_file)) as f:
            pairs = f.read().splitlines()

        num_matched, num_mismatched = int(pairs[0]), int(pairs[0])
        pairs = [p.split() for p in pairs[1:]]

        return num_matched, num_mismatched, pairs

    def __getitem__(self, index):
        pair = self.pairs[index]
        target = self.targets[index]

        filename = "{}_{:04d}.jpg"
        if index < self.num_matched:
            # matched pair
            path_1 = os.path.join(
                self.root, "lfw", pair[0], filename.format(pair[0], int(pair[1])))
            path_2 = os.path.join(
                self.root, "lfw", pair[0], filename.format(pair[0], int(pair[2])))
            img1 = Image.open(path_1)
            img2 = Image.open(path_2)

        else:
            # mismatched pair
            path_1 = os.path.join(
                self.root, "lfw", pair[0], filename.format(pair[0], int(pair[1])))
            path_2 = os.path.join(
                self.root, "lfw", pair[2], filename.format(pair[2], int(pair[3])))
            img1 = Image.open(path_1)
            img2 = Image.open(path_2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, target

    def __len__(self):
        return len(self.targets)


class PeopleLFW(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        if split == "train":
            self.txt_file = "peopleDevTrain.txt"
        elif split == "test":
            self.txt_file = "peopleDevTest.txt"
        else:
            raise ValueError(
                "Split \"{}\" is not in (\"train\", \"test\").".format(split))

        self.root = root
        classes, class_to_idx, samples = self._read_txt()
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform

    def _read_txt(self):
        with open(os.path.join(self.root, self.txt_file)) as f:
            text = f.read().splitlines()

        num_people = int(text[0])
        people = [p.split() for p in text[1:]]
        classes = [p[0] for p in people]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        samples = []
        filename = "{}_{:04d}.jpg"
        for i in range(num_people):
            num_imgs = int(people[i][1])
            for j in range(num_imgs):
                path = os.path.join(
                    self.root, "lfw", classes[i], filename.format(classes[i], j + 1))
                samples.append((path, i))

        return classes, class_to_idx, samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.targets)


class PeopleLFWPair(PeopleLFW):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class CASIAPair(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class CelebAPair(CelebA):
    def __getitem__(self, index):
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            pos_1 = self.transform(X)
            pos_2 = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return pos_1, pos_2, target


class AttGANTransform(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, attgan) -> None:
        super().__init__()
        self.attgan = attgan

    @torch.no_grad()
    def __call__(self, img):
        img = img.to(next(self.attgan.parameters()).device)
        img = img * 2 - 1
        img = img.unsqueeze(0)
        att_b = torch.rand((1, 13)) * 2 - 1
        out_img = self.attgan(img, att_b).squeeze(0)
        return (out_img + 1) / 2


def get_color_distortion(s=0.5):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


def get_simclr_pipeline_transform(args):
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    if args.random_resized_crop:
        data_transforms.append(transforms.RandomResizedCrop(size=args.img_sz))
    if args.h_flip:
        data_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    if args.color_distort:
        data_transforms.append(get_color_distortion(s=0.5))
    if args.attgan_transform:
        data_transforms.append(AttGANTransform(args.attgan))
    
    data_transforms = transforms.Compose(data_transforms)
    return data_transforms


if __name__ == '__main__':

    class ARGS:
        img_sz = 128
        attgan_transform = True
        attgan = AttGANGenerator().to("cpu")
        state_dict = torch.load("src/face-editing/attgan.pth")
        attgan.load_state_dict(state_dict)
        random_resized_crop = True
        h_flip = True
        color_distort = True

    args = ARGS()
    train_data = PeopleLFWPair(
                root='../datasets/LabeledFaceWild', split='train',
                transform=get_simclr_pipeline_transform(args))

    # train_data = CASIAPair(
    #             root='../datasets/CASIA-WebFace',
    #             transform=get_simclr_pipeline_transform())

    x = next(iter(train_data))
    plt.imshow(train_data[5][0].permute(1, 2, 0).cpu().numpy())
    plt.show()