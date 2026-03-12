import torch
import torch.nn as nn
from models.FuzzyLayer_Attention import FuzzyLayer_OR


class Encoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False, down_sample=False):
        super(Encoder3D, self).__init__()
        if down_sample:
            self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1, stride=2)
        else:
            self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            self.drop = nn.Dropout3d(p=dropout)

    def forward(self, x):
        e0 = self.conv1(x)
        e0 = self.norm1(e0)
        e0 = self.act1(e0)
        e1 = self.conv2(e0)
        e1 = self.norm2(e1)
        e1 = self.act2(e1)
        return e0, e1


class Center3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, Up_method, dropout=False):
        super(Center3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1, stride=2)
        self.norm1 = nn.InstanceNorm3d(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        if Up_method == 'ConvTrans':
            self.up = nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        elif Up_method == 'Upsampling':
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            self.drop = nn.Dropout3d(p=dropout)

    def forward(self, x):
        c = self.conv1(x)
        c = self.norm1(c)
        c = self.act1(c)
        c = self.conv2(c)
        c = self.norm2(c)
        c = self.act2(c)
        c = self.up(c)
        return c


class Decoder3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, Up_method, dropout=False):
        super(Decoder3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1, stride=1)
        self.norm1 = nn.InstanceNorm3d(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)
        if Up_method == 'ConvTrans':
            self.up = nn.ConvTranspose3d(out_channels, deconv_channels, kernel_size=2, stride=2)
        elif Up_method == 'Upsampling':
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            self.drop = nn.Dropout3d(p=dropout)

    def forward(self, x):
        c = self.conv1(x)
        c = self.norm1(c)
        c = self.act1(c)
        c = self.conv2(c)
        c = self.norm2(c)
        c = self.act2(c)
        up = self.up(c)
        return c, up


class Output_Layer(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, up_scale):
        super(Output_Layer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(middle_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()
        self.up_sample = nn.Upsample(scale_factor=up_scale, mode='trilinear', align_corners=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.out_act(out)
        out = self.up_sample(out)
        return out


class Last_Layer(nn.Module):
    def __init__(self, channels):
        super(Last_Layer, self).__init__()
        self.conv1 = nn.Conv3d(channels[0], channels[1], kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels[1])
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels[1], channels[2], kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels[2])
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv3d(channels[2], channels[3], kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.out_act(out)
        return out


class FuzzyAttention_Layer(nn.Module):
    def __init__(self, filter_d, filter_e, filter_mix, fuzzy_num):
        super(FuzzyAttention_Layer, self).__init__()
        self.conv1 = nn.Conv3d(filter_d, filter_mix, kernel_size=1)
        self.norm1 = nn.InstanceNorm3d(filter_mix)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(filter_e, filter_mix, kernel_size=1)
        self.norm2 = nn.InstanceNorm3d(filter_mix)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fuzzyattention = FuzzyLayer_OR(filter_mix,filter_mix,fuzzy_num)

    def forward(self, e, d):
        d1 = self.act1(self.norm1(self.conv1(d)))
        e1 = self.act2(self.norm2(self.conv2(e)))
        fusion = self.relu(d1+e1)
        fusion = self.fuzzyattention(fusion)
        out = e*fusion
        return out


class FuzzyAttention_3DUNet(nn.Module):
    # (16, 32, 64, 128, 256)
    def __init__(self, in_channel=1, n_classes=1, Up_method='Upsampling'):
        super(FuzzyAttention_3DUNet, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.ec0 = Encoder3D(self.in_channel, 16, 32)     # 32, w, h
        self.ec1 = Encoder3D(32, 64, 64, down_sample=True)   # 64, w/2, h/2
        self.ec2 = Encoder3D(64, 128, 128, down_sample=True)  # 128, w/4, h/4
        self.ec3 = Encoder3D(128, 256, 256, down_sample=True)  # 256, w/8, h/8
        self.c = Center3D(256,512,256, 256, Up_method)        # 256, w/8, h/8

        self.att0 = FuzzyAttention_Layer(256, 256, 256, 4)
        self.dc0 = Decoder3D(512, 128, 128, 128, Up_method)
        self.out0 = Output_Layer(128, 64, n_classes, up_scale=8)

        self.att1 = FuzzyAttention_Layer(128, 128, 128, 4)
        self.dc1 = Decoder3D(256, 64, 64, 64, Up_method)
        self.out1 = Output_Layer(64, 32, n_classes, up_scale=4)

        self.att2 = FuzzyAttention_Layer(64, 64, 64,4)
        self.dc2 = Decoder3D(128, 32, 32, 32, Up_method)
        self.out2 = Output_Layer(32, 16, n_classes, up_scale=2)

        self.att3 = FuzzyAttention_Layer(32, 32, 32,4)
        self.out = Last_Layer([64, 32, 32, self.n_classes])

    def forward(self, x):
        e0_0, e0_1 = self.ec0(x)
        e1_0, e1_1 = self.ec1(e0_1)
        e2_0, e2_1 = self.ec2(e1_1)
        e3_0, e3_1 = self.ec3(e2_1)

        center = self.c(e3_1)
        att0 = self.att0(e3_1, center)
        cat0 = torch.cat((center,att0), 1)
        dc0, up0 = self.dc0(cat0)
        out0 = self.out0(dc0)

        att1 = self.att1(e2_1, up0)
        cat1 = torch.cat((up0, att1), 1)
        dc1, up1 = self.dc1(cat1)
        out1 = self.out1(dc1)

        att2 = self.att2(e1_1, up1)
        cat2 = torch.cat((up1, att2), 1)
        dc2, up2 = self.dc2(cat2)
        out2 = self.out2(dc2)

        att3 = self.att3(e0_1, up2)
        cat3 = torch.cat((up2, att3), 1)
        out3 = self.out(cat3)

        pred = torch.cat((out0,out1,out2,out3), 1)
        return pred


def get_model():
    net = FuzzyAttention_3DUNet()
    return net

if __name__ == '__main__':
    use_gpu = True
    from torchinfo import summary
    net = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = net.to(device)
    summary(model, (1,1, 128, 96, 144))
