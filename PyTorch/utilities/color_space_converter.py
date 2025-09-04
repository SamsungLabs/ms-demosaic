import os
import numpy as np
import torch
from simple_camera_pipeline.python.pipeline_utils import get_metadata
import torch.nn as nn

class ColorSpaceConverter(nn.Module):
    def __init__(self, dng_file= os.path.join(os.path.dirname(os.path.abspath(__file__)), "DSC01201_PSMS.dng")):
        super(ColorSpaceConverter, self).__init__()
        # Load metadata from DNG file
        metadata = get_metadata(dng_file)

        self.color_correction_1 = nn.Parameter(torch.tensor([i.decimal() for i in metadata['camera_calibration_1']],
                                                            dtype=torch.float32).reshape((3, 3)), requires_grad=False)
        self.color_correction_2 = nn.Parameter(torch.tensor([i.decimal() for i in metadata['camera_calibration_2']],
                                                            dtype=torch.float32).reshape((3, 3)), requires_grad=False)
        self.forward_matrix_1 = nn.Parameter(torch.tensor([i.decimal() for i in metadata['forward_matrix_1']],
                                                          dtype=torch.float32).reshape((3, 3)), requires_grad=False)
        self.forward_matrix_2 = nn.Parameter(torch.tensor([i.decimal() for i in metadata['forward_matrix_2']],
                                                          dtype=torch.float32).reshape((3, 3)), requires_grad=False)
        self.as_shot_neutral = nn.Parameter(torch.tensor([i.decimal() for i in metadata['as_shot_neutral']],
                                                         dtype=torch.float32), requires_grad=False)
        
        self.inverse_color_correction_1 = nn.Parameter(torch.inverse(self.color_correction_1), requires_grad=False)

        # Compute the D matrix (inverse of the as_shot_neutral factors)
        D_inv = torch.tensor([[self.as_shot_neutral[0], 0, 0],
                              [0, self.as_shot_neutral[1], 0],
                              [0, 0, self.as_shot_neutral[2]]], dtype=torch.float32)
        self.D = nn.Parameter(torch.inverse(D_inv), requires_grad=False)

        # Add the xyz2srgb matrix as a non-trainable parameter
        xyz2srgb = torch.tensor([[3.1338561, -1.6168667, -0.4906146],
                                 [-0.9787684, 1.9161415, 0.0334540],
                                 [0.0719453, -0.2289914, 1.4052427]], dtype=torch.float32)
        self.xyz2srgb = nn.Parameter(xyz2srgb, requires_grad=False)
        
        #load ms_to_rgb_matrix
        
        ms_to_rgb_matrix_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../utilities/ms_to_rgb.npy")
        ms_to_rgb_matrix = np.load(ms_to_rgb_matrix_file_path)
        #make this matrix a tensor
        ms_to_rgb_matrix = torch.tensor(ms_to_rgb_matrix, dtype=torch.float32) #16x3 matrix
        self.ms_to_rgb_matrix = nn.Parameter(ms_to_rgb_matrix, requires_grad=False)
        

    def apply_color_space_transform(self, demosaiced_images):
        # Assuming demosaiced_images has shape (B, C, H, W)
        B, C, H, W = demosaiced_images.shape

        # Compute DF matrix once
        DF = torch.matmul(self.forward_matrix_1, self.D)
        DF = torch.matmul(DF, self.inverse_color_correction_1)

        # Reshape DF to be (1, 3, 3, 1, 1) for broadcasting to batch
        DF = DF.view(1, 3, 3, 1, 1)

        # Expand demosaiced_images from (B, C, H, W) to (B, 1, C, H, W) for broadcasting
        demosaiced_images = demosaiced_images.unsqueeze(1)  # (B, 1, C, H, W)

        # Multiply DF with demosaiced images
        xyz_images = torch.sum(DF * demosaiced_images, dim=2)  # Result shape (B, 3, H, W)

        return xyz_images

    def transform_xyz_to_srgb(self, xyz_images):
        # Assuming xyz_images has shape (B, 3, H, W)
        B, C, H, W = xyz_images.shape

        # Reshape xyz2srgb to be (1, 3, 3, 1, 1) for broadcasting
        xyz2srgb = self.xyz2srgb.view(1, 3, 3, 1, 1)

        # Expand xyz_images from (B, C, H, W) to (B, 1, C, H, W) for broadcasting
        xyz_images = xyz_images.unsqueeze(1)  # (B, 1, C, H, W)

        # Multiply xyz2srgb matrix with xyz_images and sum across the channel dimension
        srgb_images = torch.sum(xyz2srgb * xyz_images, dim=2)  # Result shape (B, 3, H, W)

        return srgb_images

    def forward(self, imgs, ms_input=False):
        if ms_input:
            #convert ms to rgb
            imgs = torch.clamp(torch.einsum('bchw,cm->bmhw', imgs[:,:16], self.ms_to_rgb_matrix), 0, 1)
        # imgs is expected to have shape (B, C, H, W)
        out = self.apply_color_space_transform(imgs)
        out = self.transform_xyz_to_srgb(out)
        out = torch.clamp(out, 0, 1)
        out = torch.pow(out, 1 / 2.2)  # Apply gamma correction
        out = 3 * out ** 2 - 2 * out ** 3  # Simple tone curve
        out = torch.clamp(out, 0, 1)  # Clip to [0, 1]
        return out
