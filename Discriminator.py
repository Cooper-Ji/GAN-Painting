import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # why normalize=False?
            *self.block(in_channels, 64, normalize=False),  # 3*256*256 -> 64*128*128
            *self.block(64, 128),  # 64*128*128 -> 128*64*64
            *self.block(128, 256),  # 128*64*64 -> 256*32*32
            *self.block(256, 512),  # 256*32*32 -> 512*16*16

            # Why padding first then convolution?
            nn.ZeroPad2d((1, 0, 1, 0)),  # padding left and top   512*16*16 -> 512*17*17
            nn.Conv2d(512, 1, 4, padding=1)  # 512*17*17 -> 1*16*16
        )

        self.scale_factor = 16

    @staticmethod
    # 如果在方法中不需要访问任何实例方法和属性，纯粹地通过传入参数并返回数据的功能性方法，
    # 那么它就适合用静态方法来定义，它节省了实例化对象的开销成本
    def block(in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers

    def forward(self, x):
        return self.model(x)

