from collections import OrderedDict
from tensorflow.python.framework import tensor_util
import torch
import torch.nn as nn
import torch.nn.functional as F
import pprint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


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


if __name__ == '__main__':
    path_to_pb = './generator.pb'
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph_nodes = [n for n in graph_def.node]
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

    wts = [n for n in graph_nodes if n.op=='Const']

    attgan = AttGANGenerator()
    # for name, module in attgan.named_parameters():
    #     print(name) 
    # attgan.eval()
    # for name, p in attgan.named_buffers():
    #     print(name)

    state_dict = OrderedDict()

    for i in range(5):
        state_dict["enc_layers.{}.layers.1.num_batches_tracked".format(i)] = torch.tensor(0)
    
    for i in range(4):
        state_dict["dec_layers.{}.layers.1.num_batches_tracked".format(i)] = torch.tensor(0)

    with torch.no_grad():
        for i, n in enumerate(wts):
            name = n.name
            if name.endswith('/weights'):
                w = torch.tensor(tensor_util.MakeNdarray(n.attr['value'].tensor))
                w = torch.nn.Parameter(w.permute(3, 2, 0, 1), requires_grad=False)
                if name.startswith('UNetGenc'):
                    layer = name.split('/')[1]
                    if layer == 'Conv':
                        # attgan.enc_layers[0].layers[0].weight.copy_(w)
                        state_dict["enc_layers.0.layers.0.weight"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.enc_layers[n_layer].layers[0].weight = w
                        state_dict["enc_layers.{}.layers.0.weight".format(n_layer)] = w

                if name.startswith('UNetGdec'):
                    layer = name.split('/')[1]
                    if layer == 'Conv2d_transpose':
                        # attgan.dec_layers[0].layers[0].weight.copy_(w)
                        state_dict["dec_layers.0.layers.0.weight"] = w
                    else:
                        n_layer = int(layer[-1])
                        if n_layer == 4:
                            # attgan.dec_layers[n_layer][0].weight.copy_(w)
                            state_dict["dec_layers.4.0.weight"] = w
                        else:
                            # attgan.dec_layers[n_layer].layers[0].weight.copy_(w)
                            state_dict["dec_layers.{}.layers.0.weight".format(n_layer)] = w

            if name.endswith('/biases'):
                w = torch.tensor(tensor_util.MakeNdarray(n.attr['value'].tensor))
                w = torch.nn.Parameter(w, requires_grad=False)
                # attgan.dec_layers[4][0].bias.copy_(w)
                state_dict["dec_layers.4.0.bias"] = w

            if name.endswith('/beta'):
                w = torch.tensor(tensor_util.MakeNdarray(n.attr['value'].tensor))
                w = torch.nn.Parameter(w, requires_grad=False)
                if name.startswith('UNetGenc'):
                    layer = name.split('/')[1]
                    if layer == 'Conv':
                        # attgan.enc_layers[0].layers[1].bias.copy_(w)
                        state_dict["enc_layers.0.layers.1.bias"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.enc_layers[n_layer].layers[1].bias.copy_(w)
                        state_dict["enc_layers.{}.layers.1.bias".format(n_layer)] = w

                if name.startswith('UNetGdec'):
                    layer = name.split('/')[1]
                    if layer == 'Conv2d_transpose':
                        # attgan.dec_layers[0].layers[1].bias = w
                        state_dict["dec_layers.0.layers.1.bias"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.dec_layers[n_layer].layers[1].bias = w
                        state_dict["dec_layers.{}.layers.1.bias".format(n_layer)] = w

            if name.endswith('/gamma'):
                w = torch.tensor(tensor_util.MakeNdarray(n.attr['value'].tensor))
                w = torch.nn.Parameter(w, requires_grad=False)
                if name.startswith('UNetGenc'):
                    layer = name.split('/')[1]
                    if layer == 'Conv':
                        # attgan.enc_layers[0].layers[1].weight.copy_(w)
                        state_dict["enc_layers.0.layers.1.weight"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.enc_layers[n_layer].layers[1].weight.copy_(w)
                        state_dict["enc_layers.{}.layers.1.weight".format(n_layer)] = w

                if name.startswith('UNetGdec'):
                    layer = name.split('/')[1]
                    if layer == 'Conv2d_transpose':
                        # attgan.dec_layers[0].layers[1].weight.copy_(w)
                        state_dict["dec_layers.0.layers.1.weight"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.dec_layers[n_layer].layers[1].weight.copy_(w)
                        state_dict["dec_layers.{}.layers.1.weight".format(n_layer)] = w
            
            if name.endswith('/moving_mean'):
                w = torch.tensor(tensor_util.MakeNdarray(n.attr['value'].tensor))
                # w = torch.nn.Parameter(w)
                if name.startswith('UNetGenc'):
                    layer = name.split('/')[1]
                    if layer == 'Conv':
                        # attgan.enc_layers[0].layers[1].running_mean.copy_(w)
                        state_dict["enc_layers.0.layers.1.running_mean"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.enc_layers[n_layer].layers[1].running_mean.copy_(w)
                        state_dict["enc_layers.{}.layers.1.running_mean".format(n_layer)] = w

                if name.startswith('UNetGdec'):
                    layer = name.split('/')[1]
                    if layer == 'Conv2d_transpose':
                        # attgan.dec_layers[0].layers[1].running_mean.copy_(w)
                        state_dict["dec_layers.0.layers.1.running_mean"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.dec_layers[n_layer].layers[1].running_mean.copy_(w)
                        state_dict["dec_layers.{}.layers.1.running_mean".format(n_layer)] = w

            if name.endswith('/moving_variance'):
                w = torch.tensor(tensor_util.MakeNdarray(n.attr['value'].tensor))
                # w = torch.nn.Parameter(w)
                if name.startswith('UNetGenc'):
                    layer = name.split('/')[1]
                    if layer == 'Conv':
                        # attgan.enc_layers[0].layers[1].running_var.copy_(w)
                        state_dict["enc_layers.0.layers.1.running_var"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.enc_layers[n_layer].layers[1].running_var.copy_(w)
                        state_dict["enc_layers.{}.layers.1.running_var".format(n_layer)] = w

                if name.startswith('UNetGdec'):
                    layer = name.split('/')[1]
                    if layer == 'Conv2d_transpose':
                        # attgan.dec_layers[0].layers[1].running_var.copy_(w)
                        state_dict["dec_layers.0.layers.1.running_var"] = w
                    else:
                        n_layer = int(layer[-1])
                        # attgan.dec_layers[n_layer].layers[1].running_var.copy_(w)
                        state_dict["dec_layers.{}.layers.1.running_var".format(n_layer)] = w

        attgan.load_state_dict(state_dict)


        torch.save(attgan.state_dict(), 'attgan.pth')
        # attgan.eval()
        # img = Image.open("./Adam_Sandler_0001.jpg")
        # img = img.resize((128, 128))
        # img = np.asarray(img).astype(np.float32) / 255
        # img = img * 2 - 1
        # xa_ipt = np.expand_dims(img, axis=0)
        # xa_ipt = torch.tensor(xa_ipt).permute(0, 3, 1, 2)
        # a_ipt = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]])
        # b_ipt_list = [a_ipt]  # the first is for reconstruction
        # for i in range(0):
        #     tmp = np.array(a_ipt, copy=True)
        #     tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
        #     b_ipt_list.append(tmp)

        # x_opt_list = [xa_ipt]
        # for i, b_ipt in enumerate(b_ipt_list):
        #     b__ipt = (b_ipt * 2 - 1).astype(np.float32)  # !!!
        #     if i > 0:   # i == 0 is for reconstruction
        #         b__ipt[..., i - 1] = b__ipt[..., i - 1]
        #     a_ipt = torch.tensor(a_ipt)
        #     x_opt = attgan(xa_ipt, a_ipt).permute(0, 2, 3, 1).cpu().numpy()
        #     x_opt = (x_opt.squeeze(0) + 1) / 2
        #     plt.imshow(x_opt)
        #     plt.show()
            # x_opt_list.append(x_opt)

        # sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
        # sample = np.reshape(sample, (sample.shape[0], -1, sample.shape[2] * sample.shape[3], sample.shape[4]))

        # sample = (sample.squeeze(0) + 1) / 2
        # plt.imshow(sample)
        # plt.axis('off')
        # plt.show()


        # # for name, param in attgan.named_parameters():
        # #     print(name, param.size())
        # print(x_opt_list[0])