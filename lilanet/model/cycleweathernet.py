import torch
import torch.hub as hub
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Implements LiLaNet model from
    `"Boosting LiDAR-based Semantic Labeling by Cross-Modal Training Data Generation"
    <https://arxiv.org/abs/1804.09915>`_.

    Arguments:
        num_classes (int): number of output classes
    """

    def __init__(self, input_dim, num_filter, output_dim, num_lilablock):
        super(Generator, self).__init__()

        # Reflection padding
        self.pad = torch.nn.ReflectionPad2d(3)
        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7, stride=1, padding=0)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        # Resnet blocks
        self.resnet_blocks = []
        for i in range(num_lilablock):
            self.resnet_blocks.append(LiLaBlock(num_filter * 4,num_filter * 4))
        self.resnet_blocks = torch.nn.Sequential(*self.resnet_blocks)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 4, num_filter * 2)
        self.deconv2 = DeconvBlock(num_filter * 2, num_filter)
        self.deconv3 = ConvBlock(num_filter, output_dim,
                                 kernel_size=7, stride=1, padding=0, activation='tanh', batch_norm=False)


    def forward(self, distance, reflectivity):
        x = torch.cat([distance, reflectivity], 1)
        # Encoder
        print("Generate input shape", x.shape)
        enc1 = self.conv1(self.pad(x))
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        # Resnet blocks
        res = self.resnet_blocks(enc3)
        # Decoder
        dec1 = self.deconv1(res)
        dec2 = self.deconv2(dec1)
        out = self.deconv3(self.pad(dec2))
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)
            if isinstance(m, LiLaBlock):
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
class Discriminator(nn.Module):
    """
    Implements LiLaNet model from
    `"Boosting LiDAR-based Semantic Labeling by Cross-Modal Training Data Generation"
    <https://arxiv.org/abs/1804.09915>`_.

    Arguments:
        num_classes (int): number of output classes
    """

    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        conv1 = ConvBlock(input_dim, num_filter, kernel_size=4, stride=2, padding=1, activation='lrelu', batch_norm=False)
        conv2 = ConvBlock(num_filter, num_filter * 2, kernel_size=4, stride=2, padding=1, activation='lrelu')
        conv3 = ConvBlock(num_filter * 2, num_filter * 4, kernel_size=4, stride=2, padding=1, activation='lrelu')
        conv4 = ConvBlock(num_filter * 4, num_filter * 8, kernel_size=4, stride=1, padding=1, activation='lrelu')
        conv5 = ConvBlock(num_filter * 8, output_dim, kernel_size=4, stride=1, padding=1, activation='no_act', batch_norm=False)

        self.conv_blocks = torch.nn.Sequential(
            conv1,
            conv2,
            conv3,
            conv4,
            conv5
        )

    def forward(self, distance, reflectivity):
        x = torch.cat([distance, reflectivity], 1)
        out = self.conv_blocks(x)
        return out
    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)

class LiLaBlock(nn.Module):

    def __init__(self, in_channels, n):
        super(LiLaBlock, self).__init__()

        self.branch1 = BasicConv2d(
            in_channels, n, kernel_size=(
                7, 3), padding=(
                2, 0))
        self.branch2 = BasicConv2d(in_channels, n, kernel_size=3)
        self.branch3 = BasicConv2d(
            in_channels, n, kernel_size=(
                3, 7), padding=(
                0, 2))
        self.branch4 = BasicConv2d(
            in_channels, n, kernel_size=(
                3, 3), padding=(
                1, 1), dilation=(
                2, 2))
        self.conv = BasicConv2d(n * 4, n, kernel_size=1, padding=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        output = torch.cat([branch1, branch2, branch3, branch4], 1)
        output = self.conv(output)

        return output

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, activation='relu', batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.batch_norm = batch_norm
        self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu', batch_norm=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, output_padding)
        self.batch_norm = batch_norm
        self.bn = torch.nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    num_classes, height, width = 3, 32, 400

    model = Generator(2, 32, 3, 5)  # .to('cuda')
    inp = torch.randn(5, 1, height, width)  # .to('cuda')

    out = model(inp, inp)
    D_A = Discriminator(2, 32, 1)
    print(D_A(inp,inp).shape)
    assert out.size() == torch.Size([5, num_classes, height, width])

    print('Pass size check.')
