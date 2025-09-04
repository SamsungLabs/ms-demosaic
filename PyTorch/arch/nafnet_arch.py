# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .nafnet_util import LayerNorm2d
from .nafnet_local import Local_Base

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], sr_blk_nums=[], features_out = True, level_1_extra_channels = None, level_2_extra_channels=None, level_3_extra_channels=None):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.sr_blks = nn.ModuleList()
        self.sr_ups = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        

        chan = width
        
        self.encoder_convs = nn.ModuleList()
        
        for i, num in enumerate(enc_blk_nums):
            chan_in = chan
            if i == 0 and level_1_extra_channels:
                chan_in += level_1_extra_channels
            elif i == 1 and level_2_extra_channels:
                chan_in += level_2_extra_channels
            elif i == 2 and level_3_extra_channels:
                chan_in += level_3_extra_channels
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan_in) for _ in range(num)]
                )
            )
            
            if i == 0 and level_1_extra_channels:
                self.encoder_convs.append(
                    nn.Conv2d(chan_in, chan, 3,1,1)
                )
            elif i == 1 and level_2_extra_channels:
                self.encoder_convs.append(
                    nn.Conv2d(chan_in, chan, 3,1,1)
                )
            elif i == 2 and level_3_extra_channels:
                self.encoder_convs.append(
                    nn.Conv2d(chan_in, chan, 3,1,1)
                )
            
            
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

    
        #just normal nafnet
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
        
        #add sr blocks
        for num in sr_blk_nums:
            self.sr_ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 4, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.sr_blks.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
        

        self.padder_size = 2 ** len(self.encoders)
        self.features_out = features_out

    def forward(self, inp, level_1_extra=None, level_2_extra=None, level_3_extra=None):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        i = 0
        for encoder, down in zip(self.encoders, self.downs):
            if i == 0 and level_1_extra is not None:
                x = torch.cat([x, level_1_extra], dim=1)
            elif i == 1 and level_2_extra is not None:
                x = torch.cat([x, level_2_extra], dim=1)
            elif i == 2 and level_3_extra is not None:
                x = torch.cat([x, level_3_extra], dim=1)
            x = encoder(x)
            if i == 0 and level_1_extra is not None:
                x = self.encoder_convs[i](x)
            elif i == 1 and level_2_extra is not None:
                x = self.encoder_convs[i](x)
            elif i == 2 and level_3_extra is not None:
                x = self.encoder_convs[i](x)
            encs.append(x)
        
            x = down(x)
            i+=1

        x = self.middle_blks(x)
        
        features_out = {}
        #store features
        if self.features_out:
            features_out[f"decoder_{len(features_out)}"] = x
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            #store features
            if self.features_out:
                features_out[f"decoder_{len(features_out)}"] = x
        
        #sr case
        for sr_blk, sr_up in zip(self.sr_blks, self.sr_ups):
            x = sr_up(x)
            x = sr_blk(x)
            #store features
            if self.features_out:
                features_out[f"decoder_{len(features_out)}"] = x
                

        out = self.ending(x)
        #x = x + inp

        if self.features_out:
            return out[:, :, :, :], features_out
        else:
            return out[:, :, :, :],

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)