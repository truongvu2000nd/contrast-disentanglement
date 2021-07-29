import torch
from torchvision import transforms

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
