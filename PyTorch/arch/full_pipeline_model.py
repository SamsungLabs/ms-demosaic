
import os
import numpy as np
import torch
import torch.nn as nn
from PyTorch.arch.mcan_model import MCANModel
from PyTorch.arch.nafnet_ind import NafNetIndModel
from PyTorch.arch.nafnet_arch import NAFNet
from PyTorch.arch.nafnet_sr_model import NafNetSRIndModel
from PyTorch.arch.restormer_model import RestormerModel
from .flow_utils import warp_image, load_pretrained_checkpoint
from .restormer_arch import Restormer_Special
from torchvision.models.optical_flow import raft_small
from torchvision.ops import deform_conv2d
from PyTorch.utilities.color_space_converter import ColorSpaceConverter


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, learn_offset=True, kernel_size=3, stride=1, padding=1):
        super(DeformConv2d, self).__init__()
        
        # Weights for the deformable convolution (like regular conv weights)
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size)) #randn before
        self.bias = nn.Parameter(torch.zeros(out_channels))  # Optional bias
        
        #Fill the weight with identity kernels
        if kernel_size % 2 == 1:  # Ensure kernel_size is odd
            center = kernel_size // 2
            for oc in range(out_channels):
                for ic in range(in_channels):
                    if ic == oc:  # Only set the center pixel if ic == oc
                        self.weight.data[oc, ic, center, center] = 1.0  # Set the center to 1
                    else:
                        self.weight.data[oc, ic, center, center] = 0.0
        self.learn_offset = learn_offset
        if self.learn_offset:    
            # Learnable offset bias: it should match the required shape for offsets
            self.offset_bias = nn.Parameter(torch.zeros(1, 2 * kernel_size * kernel_size, 1, 1))  # Shape: (1, 2*k*k, 1, 1)
        
    def forward(self, x, offset_flow=None):
        # Generate the offset map using the learned offset_bias with out-of-place operation
        batch_size = x.size(0)
        if self.learn_offset:
            offset = self.offset_bias.data.repeat(batch_size, 1, x.size(2), x.size(3))  # Repeat to match batch size
        
            if offset_flow is not None:
                # RAFT flow is in shape (B, 2, H, W), split into x and y flow
                flow_x = offset_flow[:, 0, :, :].unsqueeze(1)  # Shape: B x 1 x H x W
                flow_y = offset_flow[:, 1, :, :].unsqueeze(1)  # Shape: B x 1 x H x W
                
                # Adjust the offsets: flow_x to x-offsets, flow_y to y-offsets
                offset[:, 0::2, :, :] += flow_y  # Add y-flow to y-offsets (even indices)
                offset[:, 1::2, :, :] += flow_x  # Add x-flow to x-offsets (odd indices)
        else:
            #size of offset_flow is [b,9, h, w, 2]
            #reshape offset_flow to [b, 18, h, w] so that it can be used as offset_bias for deformable convolution
            offset = offset_flow.permute(0, 1, 4, 2, 3).contiguous()  # Permute to bring the last dimension (2) forward
            offset = offset.view(offset.size(0), -1, offset.size(3), offset.size(4))  # Flatten the last two dimensions (9, 2) -> (18)
            
        # Apply deformable convolution with the adjusted offset
        out = deform_conv2d(x, offset, self.weight, self.bias, stride=1, padding=1)
        
        return out

                        
class FullPipeline(nn.Module,):
    def __init__(self, backbone, in_channels=2, sr=False):
        super(FullPipeline, self).__init__()
        self.in_channels =in_channels
        self.out_channels = 16
        
        self.csc = ColorSpaceConverter()
        
        self.sr = sr
        
        dir_name = os.path.dirname(os.path.abspath(__file__))

        if self.sr:
            if backbone == "restormer":
                self.ms_backbone = RestormerModel(in_channels = 1, out_channels=16, sr=True, features_out=True)
                load_pretrained_checkpoint(self.ms_backbone, f"{dir_name}/../base_models/restormer_convnet_ms_mosaic_ds4_10-29-14-12-16/best.pth")
            elif backbone == "naf":
                self.ms_backbone = NafNetIndModel(in_channels = 1, out_channels=16, features_out = True, size="full", sr=True)
                load_pretrained_checkpoint(self.ms_backbone, f"{dir_name}/../base_models/nafms_convnet_ms_mosaic_ds4_10-29-3-29-52/best.pth")
            elif backbone == "nafsr":
                self.ms_backbone = NafNetSRIndModel(in_channels = 1, out_channels=16, features_out = True, size="full")
                load_pretrained_checkpoint(self.ms_backbone, f"{dir_name}/../base_models/nafsr_restormer_deform_ms_mosaic_ds4_11-5-1-51-28/best.pth")
            elif backbone == "mcan":
                self.ms_backbone = MCANModel(in_channels = 1, out_channels=16, sr=True, size="normal")
                load_pretrained_checkpoint(self.ms_backbone, f"{dir_name}/../base_models/mcan_restormer_deform_ms_mosaic_ds4_11-6-23-59-49/best.pth")
            
        else:
            if backbone == "restormer":
                self.ms_backbone = RestormerModel(in_channels = 1, out_channels=16, features_out=True)
                load_pretrained_checkpoint(self.ms_backbone, f"{dir_name}/../base_models/restormer_convnet_ms_mosaic_10-26-5-43-27/best.pth")
            elif backbone == "naf":
                self.ms_backbone = NafNetIndModel(in_channels = 1, out_channels=16, features_out = True, size="full")
                load_pretrained_checkpoint(self.ms_backbone, f"{dir_name}/../base_models/nafms_convnet_ms_mosaic_10-26-10-56-5/best.pth")
            elif backbone == "mcan":
                self.ms_backbone = MCANModel(in_channels = 1, out_channels=16, sr=False, size="normal")
                load_pretrained_checkpoint(self.ms_backbone, f"{dir_name}/../base_models/mcan_convnet_ms_mosaic_10-26-5-48-57/best.pth")

        self.rgb_backbone = NafNetIndModel(in_channels = 1, out_channels=3, features_out = True, size="small")
        

       
        load_pretrained_checkpoint(self.rgb_backbone, f"{dir_name}/../base_models/nafrgb_convnet_rgb_mosaic_10-26-5-50-32/best.pth")
        
        self.flow = raft_small(pretrained=True, progress=False)
     

       
        in_channels = 16+3+1
        level_1_extra_channels = 32
        level_2_extra_channels = 64
        level_3_extra_channels = 128
        
        self.deform_conv_scale_1 = DeformConv2d(32, level_1_extra_channels)
        self.deform_conv_scale_2 = DeformConv2d(64, level_2_extra_channels)
        self.deform_conv_scale_3 = DeformConv2d(128, level_3_extra_channels)
        self.c2_deform_conv_scale = DeformConv2d(64, 64, False) #this is extra (I don't use this)

        self.enhance_backbone = NAFNet(in_channels = in_channels, out_channels=self.out_channels, level_1_extra_channels = level_1_extra_channels, level_2_extra_channels=level_2_extra_channels, level_3_extra_channels=level_3_extra_channels, width=32, middle_blk_num=2, enc_blk_nums=[2,2,2], dec_blk_nums=[2,2,2])
            
        #load ms_to_rgb_matrix
        ms_to_rgb_matrix_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../utilities/ms_to_rgb.npy")
        ms_to_rgb_matrix = np.load(ms_to_rgb_matrix_file_path)
        #make this matrix a tensor
        ms_to_rgb_matrix = torch.tensor(ms_to_rgb_matrix, dtype=torch.float32) #16x3 matrix
        
        # Register the matrix as a buffer (fixed, not trainable)
        self.register_buffer('ms_to_rgb_matrix', ms_to_rgb_matrix)
        
        self.sr = sr


    #show the difference between a 16x3 matrix and using some color alignment to get the flow
        
    def forward(self, x):
        if self.sr:
            B, C, H, W = x.size()
            #get top left of ms_mosaic
            ms_mosaic = x[:, :1, :H//4, :W//4]
            rgb_mosaic = x[:, 1:, :, :]
        else: #normal size
            ms_mosaic = x[:, :1, :, :]
            rgb_mosaic = x[:, 1:, :, :]
            
        with torch.no_grad():
            ms_out, ms_features = self.ms_backbone(ms_mosaic)
            rgb_out, rgb_features = self.rgb_backbone(rgb_mosaic)
    
        #convert ms to rgb
        ms_conv_rgb = torch.clamp(torch.einsum('bchw,cm->bmhw', ms_out[:,:16]+0.5, self.ms_to_rgb_matrix), 0, 1)-0.5
        with torch.no_grad():
            flow_rgb_to_ms_pixels = self.flow(self.csc(ms_out[:,:16]+0.5, ms_input=True), self.csc(torch.clamp(rgb_out+0.5, 0, 1)))[-1] #warping from rgb to ms # Normalize images from [-0.5, 0.5] to [0, 1]
        #make flow same size as flow_pixels
        flow_rgb_to_ms = torch.empty_like(flow_rgb_to_ms_pixels)
        flow_rgb_to_ms[:,0] = (flow_rgb_to_ms_pixels[:,0] / (x.size(3))) * 2.0 #normalize flow to [-1, 1]
        flow_rgb_to_ms[:,1] = (flow_rgb_to_ms_pixels[:,1] / (x.size(2))) * 2.0 #normalize flow to [-1, 1]
            
        
        #warp the rgb image
        mask_invalid_flow_rgb, warped_rgb = warp_image(rgb_out, flow_rgb_to_ms)
    
        #get rgb_feature_keys
        rgb_features_keys = sorted(list(rgb_features.keys()))
        
        #concatenate rgb_out to rgb_features
        warped_rgb_features_decoder_4 = self.deform_conv_scale_1(rgb_features[rgb_features_keys[-1]], offset_flow=flow_rgb_to_ms_pixels)  # Pass the flow as the offset bias
        

        #scale flow to the size of the feature map (just average pool)
        flow_pixels_scale_2 = torch.nn.functional.interpolate(flow_rgb_to_ms_pixels, size=(ms_out.size(2)//2, ms_out.size(3)//2), mode='bilinear', align_corners=False).squeeze(1)
        flow_pixels_scale_2 = flow_pixels_scale_2 / 2.0
        warped_rgb_features_decoder_3 = self.deform_conv_scale_2(rgb_features[rgb_features_keys[-2]], offset_flow=flow_pixels_scale_2)  # Pass the flow as the offset bias
        
        flow_pixels_scale_4 = torch.nn.functional.interpolate(flow_rgb_to_ms_pixels, size=(ms_out.size(2)//4, ms_out.size(3)//4), mode='bilinear', align_corners=False).squeeze(1)
        flow_pixels_scale_4 = flow_pixels_scale_4 / 4.0
        warped_rgb_features_decoder_2 = self.deform_conv_scale_3(rgb_features[rgb_features_keys[-3]], offset_flow=flow_pixels_scale_4)
        
        
        # Generate a mask for invalid flow values if necessary
        mask_invalid_flow, warped_rgb = warp_image(rgb_out, flow_rgb_to_ms)
        # Concatenate the inputs for the enhance_backbone (MS features + warped RGB + mask)
        network_in = torch.cat((ms_out[:, :16], warped_rgb, mask_invalid_flow), dim=1)
        # Process the concatenated inputs through the enhancement backbonen            
        network_out = self.enhance_backbone(network_in, warped_rgb_features_decoder_4, warped_rgb_features_decoder_3, warped_rgb_features_decoder_2)
        
        if type(network_out) == tuple:
            network_out = network_out[0]
            
        return network_out, ms_out[:,:16], ms_conv_rgb, rgb_out, warped_rgb, flow_rgb_to_ms, mask_invalid_flow_rgb
    