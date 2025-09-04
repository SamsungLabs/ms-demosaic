
import torch.nn as nn
from .restormer_arch import Restormer, SmallRestormer
import torch.nn.functional as F


class RestormerModel(nn.Module):

    def __init__(self, in_channels, out_channels=3, features_out=False, size="normal", sr=False):
        super(RestormerModel, self).__init__()
        #TODO:missing kernel initializer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sr = sr

        if size == "small":
            self.restormer = SmallRestormer(in_channels=in_channels, out_channels=out_channels,  dim = 32,
                                       num_blocks=[1, 1],
                                        num_refinement_blocks=1,
                                        heads=[1, 1],
                                        sr=sr
                                        )
            
        elif size == "normal":
            self.restormer = Restormer(in_channels=in_channels, out_channels=out_channels,  dim = 48,
            num_blocks = [4,6,6,8], 
            num_refinement_blocks = 4,
            heads = [1,2,4,8],
            sr=sr,
            features_out=features_out
            )
        elif size == "large":
            self.restormer = Restormer(in_channels=in_channels, out_channels=out_channels,  dim = 48,
            num_blocks = [8,10,10,12],
            num_refinement_blocks = 8,
            heads = [1,2,4,8],
            sr=sr,
            features_out=features_out
            )
        
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.features_out = features_out
        
        
      
    
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
            x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom), mode= "replicate")
            
        # 4x pixel shuffle to restore spatial resolution
        x = self.pixel_shuffle(x) 
        # Pass through the network
        out = self.restormer(x)
        
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
        