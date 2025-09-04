import torch.nn as nn
from PyTorch.arch.nafssr_arch import NAFNetSR
import torch.nn.functional as F

class NafNetSRIndModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, features_out=False, size="full"):
        super(NafNetSRIndModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if size == "large":
            self.backbone = NAFNetSR(up_scale=4, width=128, num_blks=220, in_channels=in_channels, out_channels=out_channels, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False)
        else:
            self.backbone = NAFNetSR(up_scale=4, width=128, num_blks=128, in_channels=in_channels, out_channels=out_channels, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dual=False)
        
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
            im = out
            features = None
        
        #crop the output to the original size
        if padding_used:
            im = im[:,:,padding_top*16:-padding_bottom*16,padding_left*16:-padding_right*16]

        return im, features
        
    