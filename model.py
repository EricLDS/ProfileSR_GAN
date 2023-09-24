import os
import torch
import torch.nn as nn
import config


class PolishLoss():
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.relu = nn.ReLU()
        self.outl_pool = nn.MaxPool1d(kernel_size=config.OUTL_WS, stride=config.OUTL_ST)
        self.swit_pool = nn.AvgPool1d(kernel_size=config.SWIT_WS, stride=config.SWIT_ST)
        if config.CUDA:
            self.mse.cuda()
            self.relu.cuda()
            self.outl_pool.cuda()
            self.swit_pool.cuda()

    def get_outline(self, prfl):
        upper = self.outl_pool(prfl[:, 0, :])
        bottom = -self.outl_pool(-prfl[:, 0, :])
        return torch.cat((upper, bottom), dim=1)

    def get_switches(self, prfl):
        dea = prfl[:, 0, 1:prfl.shape[2]] - prfl[:, 0, 0:prfl.shape[2] - 1]
        mask = dea.ge(0)
        switch_points = mask[:, 0:mask.shape[1] - 1] ^ mask[:, 1:mask.shape[1]]
        return switch_points.float()

    def get_dea(self, prfl):
        dea = torch.abs(prfl[:, 0, 2:prfl.shape[2]] - prfl[:, 0, 0:prfl.shape[2] - 2])
        return dea


    def cal_loss(self, fake, real):
        f_outline = self.get_outline(fake)
        r_outline = self.get_outline(real)
        loss_outline = self.mse(self.outl_pool(f_outline),
                                self.outl_pool(r_outline)) * config.W_OUTL

        f_dea = self.get_dea(fake)
        r_dea = self.get_dea(real)
        loss_switch = self.mse(self.swit_pool(f_dea),
                               self.swit_pool(r_dea)) * config.W_SWIT

        return loss_outline, loss_switch


class Disc(nn.Module):
    def __init__(self, num_fea, name='disc'):
        super(Disc, self).__init__()
        self.name = name
        self.fea_dim = num_fea

        self.conv1 = self.conv_block(2, self.fea_dim, 4, 2, 1)
        self.conv2 = self.conv_block(self.fea_dim, self.fea_dim * 2, 4, 2, 1)
        self.conv3 = self.conv_block(self.fea_dim * 2, self.fea_dim * 4, 4, 2, 1)
        self.conv4 = self.conv_block(self.fea_dim * 4, self.fea_dim * 8, 4, 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(int(self.fea_dim * 8 * config.DIM_HR / 16), 1),
            nn.Dropout(config.DROPRATE)
        )
        self.pool = nn.MaxPool1d(kernel_size=4, stride=2)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def build_input(self, hr, bs):
        derivative = torch.zeros(bs, 1, config.DIM_HR).to(config.DEVICE)
        derivative[:, 0, 0:config.DIM_HR - 1] = hr[:, 0, 1:config.DIM_HR] \
                                                - hr[:, 0, 0:config.DIM_HR - 1]
        derivative[:, 0, config.DIM_HR - 1] = hr[:, 0, config.DIM_HR - 2]
        return torch.cat((hr, derivative), dim=1)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            # nn.Dropout(config.DROPRATE)
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = torch.flatten(x4, start_dim=1)
        out = torch.sigmoid(self.fc(x5))
        return out, x5

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + self.name + config.TAG + '_best.h5')
        torch.save(self.state_dict(), filename)


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            discriminator=False,
            use_act=True,
            use_bn=True,
            **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv1d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )
        # self.drop = nn.Dropout(config.DROPRATE)

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv1d(in_c, in_c * scale_factor, 5, 1, 2)
        self.ps = PixelShuffle1D(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Gen15to5_SRGAN_norm(nn.Module):
    def __init__(self, in_channels=6, num_channels=64, num_blocks=16, name='Gen'):
        super().__init__()
        self.name = name
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=5, stride=1, padding=2, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 3))
        self.final = nn.Conv1d(num_channels, 1, kernel_size=9, stride=1, padding=4)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return self.sigm(self.final(x)) * config.P_MAX

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + self.name + config.TAG + '_best.h5')
        torch.save(self.state_dict(), filename)


class Gen30to5_SRGAN_norm(nn.Module):
    def __init__(self, in_channels=6, num_channels=64, num_blocks=16, name='Gen'):
        super().__init__()
        self.name = name
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=5, stride=1, padding=2, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 3))
        self.final = nn.Conv1d(num_channels, 1, kernel_size=9, stride=1, padding=4)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return self.sigm(self.final(x)) * config.P_MAX

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + self.name + config.TAG + '_best.h5')
        torch.save(self.state_dict(), filename)


class Gen60to5_SRGAN_norm(nn.Module):
    def __init__(self, in_channels=2, num_channels=64, num_blocks=16, name='Gen'):
        super().__init__()
        self.name = name
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=5, stride=1, padding=2, use_act=False)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2),
                                       UpsampleBlock(num_channels, 2),
                                       UpsampleBlock(num_channels, 3))
        self.final = nn.Conv1d(num_channels, 1, kernel_size=9, stride=1, padding=4)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return self.sigm(self.final(x)) * config.P_MAX

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + self.name + config.TAG + '_best.h5')
        torch.save(self.state_dict(), filename)


class Gen5min_POL_norm(nn.Module):
    def __init__(self, in_channels=1, num_channels=64, num_blocks=16, name='Gen'):
        super().__init__()
        self.name = name
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=5, stride=1, padding=2, use_act=False)
        self.final = nn.Conv1d(num_channels, 1, kernel_size=9, stride=1, padding=4)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        return self.sigm(self.final(x)) * config.P_MAX

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + self.name + config.TAG + '_best.h5')
        torch.save(self.state_dict(), filename)


class Gen5min_POL(nn.Module):
    def __init__(self, in_channels=1, num_channels=64, num_blocks=16, name='Gen'):
        super().__init__()
        self.name = name
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=5, stride=1, padding=2, use_act=False)
        self.final = nn.Conv1d(num_channels, 1, kernel_size=9, stride=1, padding=4)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        return self.final(x)

    def save_checkpoint(self, epoch):
        if epoch % config.SAVE_PER_EPO == 0:
            filename = os.path.join("../checkpoint/" + config.TAG + '/' + self.name + '_epoch' + str(epoch) + '.h5')
            torch.save(self.state_dict(), filename)

    def save_best_checkpoint(self):
        filename = os.path.join("../checkpoint/" + self.name + config.TAG + '_best.h5')
        torch.save(self.state_dict(), filename)

