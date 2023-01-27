import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=256*8*8, out_features=10)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = x.view(-1, 256*8*8)
        x = self.fc(x)
        return x

class ConvM(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        norm_layer = nn.BatchNorm2d
        super(ConvM, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True),
        )
        
class ConvNet(nn.Module):
    def __init__(self, n_class=3):
        super(ConvNet, self).__init__()
        
        self.conv = nn.Sequential(
            ConvM(3, 32, 5, 2),
            ConvM(32, 64, 5, 2),
            ConvM(64, 128, 3, 1),
            ConvM(128, 64, 3, 1),
            ConvM(64, 32, 3, 1),
        )        
        self.fc = nn.Linear(32, n_class)
    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        ft = x
        output = self.fc(x)
        return output

class MLP(nn.Module):
    def __init__(self, n_class=3):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(3*28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128,n_class)
        self.droput = nn.Dropout(0.2)
        
    def forward(self,x):
        x = x.view(-1,3*28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.droput(x)
        # x = self.fc4(x)
        return x

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class ResnetEncoder(nn.Module):
    # 202005251539 attr dim
    def __init__(self, input_nc=3, output_nc=3, n_blocks=3): 
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        ngf = 64
        padding_type ='reflect'
        norm_layer = nn.InstanceNorm2d
        use_bias = False
        
        model = [nn.Conv2d(input_nc , ngf, kernel_size=7, padding=3,
                           bias=use_bias),
                 norm_layer(ngf, affine=True, track_running_stats=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2, affine=True, track_running_stats=True),
                      nn.ReLU(True)]
        mult = 2**n_downsampling
        
        for i in range(n_blocks):
            model += [ResidualBlock(dim_in=ngf * mult, dim_out=ngf * mult)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        #a = a.view(a.size(0), a.size(1), 1, 1)
        #a = a.repeat(1, 1, input.size(2), input.size(3))
        #input = torch.cat([input, a], dim=1)
        return self.model(input)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class RESNET18(nn.Module):
    def __init__(self, num_classes=10):
        super(RESNET18, self).__init__()
        # _resnet('resnet18', BasicBlock, [2, 2, 2, 2])
        channel_ration = 0.1

        norm_layer = nn.BatchNorm2d
        outch =  int(64 * channel_ration)
        self.conv1 = nn.Conv2d(3, outch, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(outch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        '''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        '''

        # layer 1
        outch =  int(64 * channel_ration)
        self.l1_p1_conv1 = conv3x3(outch, outch, 1)
        self.l1_p1_bn1 = norm_layer(outch)
        self.l1_relu = nn.ReLU(inplace=True)
        self.l1_p1_conv2 = conv3x3(outch, outch)
        self.l1_p1_bn2 = norm_layer(outch)
        
        self.l1_p2_conv1 = conv3x3(outch, outch, 1)
        self.l1_p2_bn1 = norm_layer(outch)
        self.l1_p2_relu = nn.ReLU(inplace=True)
        self.l1_p2_conv2 = conv3x3(outch, outch)
        self.l1_p2_bn2 = norm_layer(outch)
        
         # layer 2
        inch =  int(64 * channel_ration)
        outch =  int(128 * channel_ration)
        self.downsample2 = nn.Sequential(
                conv1x1(inch, outch, 2),
                norm_layer(outch),
            )
        self.l2_p1_conv1 = conv3x3(inch, outch, 2)
        self.l2_p1_bn1 = norm_layer(outch)
        self.l2_relu = nn.ReLU(inplace=True)
        self.l2_p1_conv2 = conv3x3(outch, outch)
        self.l2_p1_bn2 = norm_layer(outch)
        
        self.l2_p2_conv1 = conv3x3(outch, outch, 1)
        self.l2_p2_bn1 = norm_layer(outch)
        self.l2_p2_relu = nn.ReLU(inplace=True)
        self.l2_p2_conv2 = conv3x3(outch, outch)
        self.l2_p2_bn2 = norm_layer(outch)
        
        # layer 3
        inch =  int(128 * channel_ration)
        outch =  int(256 * channel_ration)
        self.downsample3 = nn.Sequential(
                conv1x1(inch, outch, 2),
                norm_layer(outch),
            )    
        self.l3_p1_conv1 = conv3x3(inch, outch, 2)
        self.l3_p1_bn1 = norm_layer(outch)
        self.l3_relu = nn.ReLU(inplace=True)
        self.l3_p1_conv2 = conv3x3(outch, outch)
        self.l3_p1_bn2 = norm_layer(outch)
        
        self.l3_p2_conv1 = conv3x3(outch, outch, 1)
        self.l3_p2_bn1 = norm_layer(outch)
        self.l3_p2_relu = nn.ReLU(inplace=True)
        self.l3_p2_conv2 = conv3x3(outch, outch)
        self.l3_p2_bn2 = norm_layer(outch)
        
        # layer 4
        inch =  int(256 * channel_ration)
        outch =  int(512 * channel_ration)
        self.downsample4 = nn.Sequential(
                conv1x1(inch, outch, 2),
                norm_layer(outch),
            )     
        self.l4_p1_conv1 = conv3x3(inch, outch, 2)
        self.l4_p1_bn1 = norm_layer(outch)
        self.l4_relu = nn.ReLU(inplace=True)
        self.l4_p1_conv2 = conv3x3(outch, outch)
        self.l4_p1_bn2 = norm_layer(outch)
        
        self.l4_p2_conv1 = conv3x3(outch, outch, 1)
        self.l4_p2_bn1 = norm_layer(outch)
        self.l4_p2_relu = nn.ReLU(inplace=True)
        self.l4_p2_conv2 = conv3x3(outch, outch)
        self.l4_p2_bn2 = norm_layer(outch)
   
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(outch , num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):    
        # 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
                
        # 2,3
        identity11 = x
        x = self.l1_p1_conv1(x)
        x = self.l1_p1_bn1(x)
        x = self.l1_relu(x)
        x = self.l1_p1_conv2(x)
        x = self.l1_p1_bn2(x)
        x += identity11
        x = self.l1_relu(x)
        # 4,5
        identity12 = x
        x = self.l1_p2_conv1(x)
        x = self.l1_p2_bn1(x)
        x = self.l1_p2_relu(x)
        x = self.l1_p2_conv2(x)
        x = self.l1_p2_bn2(x)
        x += identity12
        x = self.l1_p2_relu(x)
        
        
        # 6,7
        identity21 = self.downsample2(x)
        x = self.l2_p1_conv1(x)
        x = self.l2_p1_bn1(x)
        x = self.l2_relu(x)
        x = self.l2_p1_conv2(x)
        x = self.l2_p1_bn2(x)
        x += identity21
        x = self.l2_relu(x)
        # 8,9
        identity22 = x
        x = self.l2_p2_conv1(x)
        x = self.l2_p2_bn1(x)
        x = self.l2_p2_relu(x)
        x = self.l2_p2_conv2(x)
        x = self.l2_p2_bn2(x)
        x += identity22
        x = self.l2_p2_relu(x)
        
        
        # 10,11
        identity31 = self.downsample3(x)
        x = self.l3_p1_conv1(x)
        x = self.l3_p1_bn1(x)
        x = self.l3_relu(x)
        x = self.l3_p1_conv2(x)
        x = self.l3_p1_bn2(x)
        x += identity31
        x = self.l3_relu(x)
        # 12,13
        identity32 = x
        x = self.l3_p2_conv1(x)
        x = self.l3_p2_bn1(x)
        x = self.l3_p2_relu(x)
        x = self.l3_p2_conv2(x)
        x = self.l3_p2_bn2(x)
        x += identity32
        x = self.l3_p2_relu(x)
        
                # 14,15
        identity41 = self.downsample4(x)
        x = self.l4_p1_conv1(x)
        x = self.l4_p1_bn1(x)
        x = self.l4_relu(x)
        x = self.l4_p1_conv2(x)
        x = self.l4_p1_bn2(x)
        x += identity41
        x = self.l4_relu(x)
        
        # 16,17
        identity42 = x
        x = self.l4_p2_conv1(x)
        x = self.l4_p2_bn1(x)
        x = self.l4_p2_relu(x)
        x = self.l4_p2_conv2(x)
        x = self.l4_p2_bn2(x)
        x += identity42
        x = self.l4_p2_relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

