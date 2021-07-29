import torch
from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        # layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride,
        #                     padding=padding)]
        # layers = add_normalization_2d(layers, norm_fn, n_out)
        # layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(
            nn.Conv2d(n_in, n_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(n_out, eps=1e-3, momentum=None),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.layers(x)


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        # layers = [nn.ConvTranspose2d(
        #     n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn == 'none'))]
        # layers = add_normalization_2d(layers, norm_fn, n_out)
        # layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(n_in, n_out, kernel_size,
                               stride, padding, bias=False),
            nn.BatchNorm2d(n_out, eps=1e-3, momentum=None),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class AttGANGenerator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=1, img_size=128):
        super(AttGANGenerator, self).__init__()

        MAX_DIM = 1024

        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128

        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers.append(Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1))
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)

        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers.append(ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1))
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            n_in, 3, (4, 4), stride=2, padding=1),
                        nn.Tanh(),
                    ))
        self.dec_layers = nn.ModuleList(layers)

    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs

    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1,
                                                    1, self.f_size, self.f_size)
        z = torch.cat([zs[-1], a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
                z = torch.cat([z, a_tile], dim=1)
        return z

    def forward(self, x, a=None, mode='enc-dec'):
        if a == None:
            a = torch.zeros(2, 13)
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)