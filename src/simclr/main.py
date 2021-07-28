"""main.py"""

import argparse
import numpy as np
import torch
import pprint

from solver import Solver


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)


def main(args):
    pprint.pprint(args)
    solver = Solver(args)
    if not args.test:
        solver.train()

    else:
        print(solver.evaluate())

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Contrastive Face Verification')

    parser.add_argument('--name', default='main', type=str,
                        help='name of the experiment')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--n_epochs', default=1, type=int, help='num epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--wd', default=1e-6, type=float,
                        help='weight decay')

    parser.add_argument('--dset_dir', default='data',
                        type=str, help='dataset directory')
    parser.add_argument('--dataset', default='LFW', choices=['LFW', 'DFW', 'CASIA', 'CelebA'],
                        type=str, help='dataset name')
    parser.add_argument('--img_sz', default=128, type=int, help='image size')

    parser.add_argument('--attgan_transform', default=False, type=str2bool, help='using attgan')
    parser.add_argument('--attgan_dir', default=None, type=str, help='attgan pretrained dir')
    parser.add_argument('--random_resized_crop', default=True, type=str2bool, help='using random resized crop')    
    parser.add_argument('--h_flip', default=True, type=str2bool, help='using horizontal flip')
    parser.add_argument('--color_distort', default=True, type=str2bool, help='using color distortion')

    parser.add_argument('--num_workers', default=2,
                        type=int, help='dataloader num_workers')
    parser.add_argument("--base_encoder", default='resnet18',
                        type=str, help="Base encoder")
    parser.add_argument("--feature_dim", default=512,
                        type=int, help="Feature dim")
    parser.add_argument("--projection_dim", default=128,
                        type=int, help="Projection dim")

    parser.add_argument("--temperature", default=0.5,
                        type=float, help="Temperature contrastive loss")

    parser.add_argument('--log_dir', default='logs',
                        type=str, help='log directory')
    parser.add_argument('--ckpt_dir', default='checkpoints',
                        type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None,
                        type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_save_epoch', default=10,
                        type=int, help='checkpoint save epoch')

    parser.add_argument('--output_dir', default='outputs',
                        type=str, help='output directory')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
