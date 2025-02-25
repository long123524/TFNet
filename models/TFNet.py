import torch
from torch import nn
from .FastSAM.fastsam import FastSAM
from torch.nn import functional as F
from typing import Dict, List
from utils.misc import initialize_weights


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Cross_module(nn.Module):
    # codes derived from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(Cross_module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()

        q1 = self.query_conv1(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width * height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)

        energy1 = torch.bmm(q1, k1)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v1, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1 * out1

        return out1

class BasicConv2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)          #加的
        return x

##Boundary prior-guided module (BPM)
class BPM(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(BPM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = BasicConv2(in_channel, out_channel, 1)  ##1
        self.conv1 = BasicConv2(in_channel, out_channel, 1)
        self.conv2 = BasicConv2(in_channel, out_channel, 1)
        self.cross = Cross_module(64)

    def forward(self, boundary_semantic, field_semantic):
        enhance_feature = self.sigmoid(self.conv1(boundary_semantic))*self.conv2(field_semantic) + self.conv2(field_semantic)
        enhance_features = self.cross(enhance_feature)

        return enhance_features

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class bounddary_decoder(nn.Module):
    def __init__(self, in_channel, out_channel, num_class):
        super(bounddary_decoder, self).__init__()
        self.cbam = CBAM(in_channel)
        self.c1 = BasicConv2(in_channel, out_channel, 3, 1, 1, 1)
        self.c2 = BasicConv2(in_channel, out_channel, 3, 1, 3, 3)
        self.c3 = BasicConv2(in_channel, out_channel, 3, 1, 5, 5)
        self.out_feature = nn.Sequential(
            BasicConv2(in_channel, 64, 3, 1, 1),
            nn.Conv2d(64, 1, num_class)
        )

    def forward(self, features):
        features = self.cbam(features)
        x1 = self.c1(features)
        x1 = x1+features
        x2 = self.c2(x1)
        x2 = x2+x1
        x3 = self.c3(x2)
        x3 = x3 + x2
        boundary_feature = self.out_feature(x3)

        return boundary_feature

##multibranch parallel fusion module (MPFM)
class MPFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MPFM, self).__init__()

        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(size=(64, 64), mode='bilinear')
        )
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(size=(64, 64), mode='bilinear')
        )
        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(size=(64, 64), mode='bilinear')
        )
        self.conv3 = BasicConv2(in_channels, out_channels, 3, 1, 1)
        self.conv5 = BasicConv2(in_channels, out_channels, 5, 1, 2)
        self.conv7 = BasicConv2(in_channels, out_channels, 7, 1, 3)
        ##
        self.d1 = BasicConv2(in_channels, out_channels, 3, 1, 1, 1)
        self.d3 = BasicConv2(in_channels, out_channels, 3, 1, 3, 3)
        self.d5 = BasicConv2(in_channels, out_channels, 3, 1, 5, 5)

        ##
        self.conv1 = BasicConv2(out_channels*3, out_channels, kernel_size=1)

        ###
        # self.conc1 = nn.Conv2d(out_channels*3, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        ##benach 1
        out1 = self.pool1(x)
        out1 = out1 +x
        out2 = self.pool2(out1)
        out2 = out1+out2
        out3 = self.pool3(out2)
        out3 = out2+out3
        ##分支二
        conv_out = self.conv3(x)
        conv_out = conv_out+x
        conv_out1 = self.conv5(conv_out)
        conv_out1 = conv_out1+conv_out
        conv_out2 = self.conv7(conv_out1)
        conv_out2 = conv_out2+conv_out1
        ##分支三
        dout = self.d1(x)
        dout = dout+x
        dout3 = self.d3(dout)
        dout3 = dout3+dout
        dout5 = self.d5(dout3)
        dout5 = dout5 + dout3

        out = torch.concat((out3, conv_out2, dout5), dim=1)
        out = self.conv1(out)
        return out



class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)        
        return x

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class TFNet(nn.Module):
    def __init__(
        self,
        num_embed=8,
        model_name: str='FastSAM-x.pt',
        device: str='cuda',
        conf: float=0.4,
        iou: float=0.9,
        imgsz: int=1024,
        retina_masks: bool=True,
        done_warmup: bool=True,
        ):
        super(TFNet, self).__init__()
        self.model = FastSAM(model_name)
        self.device = device
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.image = None
        self.image_feats = None
        self.drops = nn.Dropout2d(0.4)
         
        self.Adapter32 = nn.Sequential(nn.Conv2d(640, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(128), nn.ReLU())
        self.Adapter16 = nn.Sequential(nn.Conv2d(640, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(128), nn.ReLU())
        self.Adapter8 = nn.Sequential(nn.Conv2d(320, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(128), nn.ReLU())
        self.Adapter4 = nn.Sequential(nn.Conv2d(160, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(128), nn.ReLU())

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.convb = BasicConv2(128, 128, 3, 1, 1)
        self.conva = BasicConv2(128, 128, 3, 1, 1)
        self.bpm = BPM(128, 64)
        self.mpfm = MPFM(64, 64)
        self.FB_out = bounddary_decoder(128, 128, 1)
        self.final_2 = nn.Sequential(
            BasicConv2(64, 64, 3, 1, 1),
            nn.Conv2d(64, 1, 1)
        )

        for param in self.model.model.parameters():
            param.requires_grad = False
        initialize_weights(self.Adapter32, self.Adapter16, self.Adapter8, self.Adapter4, \
                           self.up1, self.up2, self.up3, self.convb, \
                           self.conva, self.bpm, self.mpfm, self.FB_out, self.final_2)

    def run_encoder(self, image):
        self.image = image
        feats = self.model(
            self.image,
            device=self.device,
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
            )
        return feats

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor):
    
        input_shape = x1.shape[-2:]
        featsA = self.run_encoder(x1)
        ##shared_layer
        featA_s4 = self.Adapter4(featsA[3].clone())
        featA_s8 = self.Adapter8(featsA[0].clone())
        featA_s4 = self.drops(featA_s4)
        featA_s8 = self.drops(featA_s8)
        new_boundary = self.up1(featA_s8)
        F_B = self.convb(new_boundary+featA_s4)

        featA_s16 = self.Adapter16(featsA[1].clone())
        featA_s32 = self.Adapter32(featsA[2].clone())
        featA_s16 = self.drops(featA_s16)
        featA_s32 = self.drops(featA_s32)

        s32_16 = self.up2(featA_s32)
        F_A = self.conva(s32_16+featA_s16)
        F_A = self.up3(F_A)

        feature_field = self.bpm(F_B, F_A)
        out_feature = self.mpfm(feature_field)
        out_field = self.final_2(out_feature)
        out_boundary = self.FB_out(F_B)

        
        return F.interpolate(out_field, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(out_boundary, input_shape, mode="bilinear", align_corners=True)






