import torch
import torch.nn as nn
from PyTorch.arch.mcan_arch import MCANNet


def input_matrix_wpn(inH, inW, add_id_channel=False):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''

    outH, outW = inH, inW
    # h_offset = torch.ones(inH, 1, 1)
    # w_offset = torch.ones(1, inW, 1)
    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    h_offset_coord[0::4, :, 0] = 0.25
    h_offset_coord[1::4, :, 0] = 0.5
    h_offset_coord[2::4, :, 0] = 0.75
    h_offset_coord[3::4, :, 0] = 1.0

    w_offset_coord[:, 0::4, 0] = 0.25
    w_offset_coord[:, 1::4, 0] = 0.5
    w_offset_coord[:, 2::4, 0] = 0.75
    w_offset_coord[:, 3::4, 0] = 1.0

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1,2)

    return pos_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW

def mosaic_to_16_channel_torch(mosaic_batch):
    """
    Converts a batch of 1-channel mosaic images (B*1*H*W) into 16-channel images (B*16*H*W),
    where each channel corresponds to a specific pixel in the 4x4 mosaic grid.
    Empty positions in each channel are filled with zeros.
    """
    B, _, H, W = mosaic_batch.shape
    assert H % 4 == 0 and W % 4 == 0, "Image dimensions must be divisible by 4."
    
    # Initialize a 16-channel empty tensor with zeros
    channels_batch = torch.zeros((B, 16, H, W), device=mosaic_batch.device, dtype=mosaic_batch.dtype)
    
    # Fill each channel with its respective mosaic pattern value
    for i in range(4):
        for j in range(4):
            channel_idx = i * 4 + j
            channels_batch[:, channel_idx, i::4, j::4] = mosaic_batch[:, 0, i::4, j::4]
    
    return channels_batch

class SimpleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleUpsample, self).__init__()
        
        # First convolution increases channels by 4 for 2x upsampling
        self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)  # First 2x upsampling

        # Second convolution increases channels by 4 again for another 2x upsampling
        self.conv2 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)  # Second 2x upsampling
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)             
        x = self.pixel_shuffle1(x)     # First 2x upsampling
        x = self.conv2(x)              
        x = self.pixel_shuffle2(x)     # Second 2x upsampling
        x= self.conv3(x)
        return x
    
class MCANModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, sr=False, size="normal"):
        super(MCANModel, self).__init__()
        self.sr = sr
        if sr:
            self.upconv = SimpleUpsample(64, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.backbone = MCANNet(size=size)

        #self.backbone = NAFNet(in_channels = self.in_conv_channels, out_channels=self.out_conv_channels, width=64, middle_blk_num=3, enc_blk_nums=[3,3], dec_blk_nums=[3,3])
        
        
    def forward(self, x):
        
        #build input matrix
        pos_mat = input_matrix_wpn(x.size(2), x.size(3)).to(x.device)
        
        sixteen_channel_in = mosaic_to_16_channel_torch(x).to(x.device)
        
        
        convt_br1_temp, out = self.backbone([sixteen_channel_in, x], pos_mat)
        
        if self.sr:
            out = self.upconv(convt_br1_temp)
        
        return out, ""
    