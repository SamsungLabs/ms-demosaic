
import torch.nn as nn
from PyTorch.arch.nafnet_arch import NAFNet
import torch.nn.functional as F

class NafNetIndModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features_out=False, size="full", sr=False):
        super(NafNetIndModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if sr:
            sr_blks = [1,1]
        else:
            sr_blks = []
        if size=="small":
            self.backbone = NAFNet(in_channels = self.in_channels, out_channels=self.out_channels, width=32, middle_blk_num=2, enc_blk_nums=[2,2], dec_blk_nums=[2,2], sr_blk_nums=sr_blks, features_out=features_out)
        elif size == "medium":
            self.backbone = NAFNet(in_channels = self.in_channels, out_channels=self.out_channels, width=32, middle_blk_num=4, enc_blk_nums=[2,2,4], dec_blk_nums=[2,2,2], sr_blk_nums=sr_blks, features_out=features_out)
        elif size == "full":
            self.backbone = NAFNet(in_channels = self.in_channels, out_channels=self.out_channels, width=32, middle_blk_num=12, enc_blk_nums=[2,2,4,8], 
                                   dec_blk_nums=[2,2,2,2], sr_blk_nums=sr_blks, features_out=features_out)
        elif size == "large":
            self.backbone = NAFNet(in_channels = self.in_channels, out_channels=self.out_channels, width=32, middle_blk_num=16, enc_blk_nums=[6,6,8,12], 
                                   dec_blk_nums=[6,6,6,6], sr_blk_nums=sr_blks, features_out=features_out)

        #self.backbone = NAFNet(in_channels = self.in_conv_channels, out_channels=self.out_conv_channels, width=64, middle_blk_num=3, enc_blk_nums=[3,3], dec_blk_nums=[3,3])
        
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        
    def forward(self, x):
        
        # 4x pixel unshuffle
        x = self.pixel_unshuffle(x)

        padding_used = False
        if x.shape[-1] % 4 != 0 or x.shape[-2] % 4 != 0:
            padding_used = True
            pad_x = (4 - (x.shape[-1] % 4)) % 4
            pad_y = (4 - (x.shape[-2] % 4)) % 4
            padding_left = pad_x // 2
            padding_right = pad_x - padding_left
            padding_top = pad_y // 2
            padding_bottom = pad_y - padding_top
            x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode='replicate')
            
        # 4x pixel shuffle to restore spatial resolution
        x = self.pixel_shuffle(x) 
        # Pass through the network
        out = self.backbone(x)
        if len(out) == 2 and type(out) is tuple: #features
            im = out[0]
            features = out[1]
        else:
            im, = out
            features = None
        
        #crop the output to the original size
        if padding_used:
            im = im[:,:,padding_top*16:-padding_bottom*16,padding_left*16:-padding_right*16]

        if features is not None:
            return im, features
        else:
            return im,
        
    