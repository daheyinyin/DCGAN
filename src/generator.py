"""dcgan generator"""
from mindspore import nn

from src.cell import Normal
from src.config import dcgan_imagenet_cfg as cfg


def convt(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="pad"):
    weight_init = Normal(mean=0, sigma=0.02)
    return nn.Conv2dTranspose(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              weight_init=weight_init, has_bias=False, pad_mode=pad_mode)


def bm(num_features):
    gamma_init = Normal(mean=1, sigma=0.02)
    return nn.BatchNorm2d(num_features=num_features, gamma_init=gamma_init)


class Generator(nn.Cell):
    """
    DCGAN Generator
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.SequentialCell()
        #  input is Z, going into a convolution
        self.generator.append(convt(cfg.latent_size, cfg.feature_size * 8, 4, 1, 0))
        self.generator.append(bm(cfg.feature_size * 8))
        self.generator.append(nn.ReLU())
        # state size. 512 x 4 x 4
        self.generator.append(convt(cfg.feature_size * 8, cfg.feature_size * 4, 4, 2, 1))
        self.generator.append(bm(cfg.feature_size * 4))
        self.generator.append(nn.ReLU())
        # state size. 256 x 8 x 8
        self.generator.append(convt(cfg.feature_size * 4, cfg.feature_size * 2, 4, 2, 1))
        self.generator.append(bm(cfg.feature_size * 2))
        self.generator.append(nn.ReLU())
        # state size. 128 x 16 x 16
        self.generator.append(convt(cfg.feature_size * 2, cfg.channel_size, 4, 2, 1))
        self.generator.append(nn.Tanh())
        # state size. 3 x 32 x  32

    def construct(self, x):
        return self.generator(x)
