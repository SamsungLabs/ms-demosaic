"""
 Loss function
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import cv2
import torch
import numpy as np
import wandb
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim

def spectral_angle_mapping_per_image(image1, image2):
    """
    Compute the spectral angle mapping per pixel and sum the values across all pixels,
    then return the sum for each image in the batch.

    Parameters:
        image1 (array-like): The first image with shape B*C*H*W.
        image2 (array-like): The second image with shape B*C*H*W.

    Returns:
        array-like: An array containing the sum of spectral angles between corresponding
        pixels of the two images for each image in the batch.
    """
    # Reshape images to have shape (B, C, -1)
    B, C, H, W = image1.shape
    image1_flat = image1.reshape(B, C, -1)
    image2_flat = image2.reshape(B, C, -1)
    
    # Compute dot product and magnitudes
    dot_product = np.sum(image1_flat * image2_flat, axis=(1))
    magnitude1 = np.sqrt(np.sum(image1_flat ** 2, axis=(1)))
    magnitude2 = np.sqrt(np.sum(image2_flat ** 2, axis=(1)))
    
    # Compute angle
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))
    
    # Sum the angles across all pixels
    summed_angle = np.sum(angle, axis=(1))

    summed_angle = summed_angle/(C*H*W)
    
    return summed_angle


class ssim_per_image():
    @staticmethod
    def compute(output, target):
        output = output.permute(0,2,3,1)
        target = target.permute(0,2,3,1)
    
        ssim_per_image = torch.zeros(output.shape[0])
        for i in range(output.shape[0]):
            ssim_per_image[i] = float(ssim(output[i].cpu().numpy(), target[i].cpu().numpy(), multichannel=True, channel_axis=2, data_range=1))
        return output.shape[0], torch.sum(ssim_per_image), ssim_per_image
    
class mse_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.square(output - target))/output.shape[0]
        return loss
class mse_loss_w_mask():

    @staticmethod
    def compute(output, target, mask):
        
        og_shape = output.shape

        #extend mask to all channels
        mask = mask.repeat(1, output.shape[1], 1, 1)
        
        output = output[~mask]
        target = target[~mask]
        loss = torch.sum(torch.square(output - target))
        # Calculate the scaling factor
        scaling_factor = mask.numel()/(~mask).sum() 
        # Rescale the loss
        loss = loss * scaling_factor /og_shape[0]
        
        return loss

class psnr_per_image():
    @staticmethod
    def compute(output, target):
        mse_per_image = torch.sum(torch.square(output - target), dim=(1,2,3))/(output.shape[1]*output.shape[2]*output.shape[3])
        MAX = 1
        psnrs_per_image = 10*(torch.log10(MAX**2/mse_per_image))
        return output.shape[0], torch.sum(psnrs_per_image), psnrs_per_image
    
class psnr_per_image_masked():
    @staticmethod
    def compute(output, target, mask=None):
        s = 4  # Assuming the mosaic pattern size is 4x4
        
        def replace_invalid_masks(mask_in):
            # Get the dimensions of the mask
            N, _, H, W = mask_in.shape
            
            # Loop through each image in the batch
            for i in range(N):
                # Check if all values in the mask are False
                if ~mask_in[i].any():
                    # Create a new mask: 
                    new_mask = torch.ones((H, W), dtype=torch.bool)
                    mask_in[i] = new_mask.unsqueeze(0)  # Assign back to batch
            
            return mask_in
        
        mask = replace_invalid_masks(mask)


        if mask is None:
            # Create a mask to exclude the pixels that the model knows
            mask = torch.ones_like(output, dtype=torch.bool)
            for i in range(s):
                for j in range(s):
                    mask[:, i*s + j, i::s, j::s] = False  # Set known pixels to False in the mask
        else:
            mask = mask.repeat(1, output.shape[1], 1, 1)
        
        #empty tensor to store psnr values
        psnrs_per_image = torch.zeros(output.shape[0])
        #compute psnr for each image
        for i in range(output.shape[0]):
            masked_output = output[i, mask[i]]
            masked_target = target[i, mask[i]]
            
            
            # Compute MSE on the unmasked (unknown) pixels
            mse_per_image = torch.sum(torch.square(masked_output - masked_target)) / masked_output.shape[0]
            MAX = 1
            psnrs_per_image[i] = 10 * (torch.log10(MAX**2 / mse_per_image))
            
        return output.shape[0], torch.sum(psnrs_per_image), psnrs_per_image
    

class mae_loss():
    @staticmethod
    def compute(output, target):
        loss = torch.sum(output - target)/output.shape[0]
        return loss
    
class mse_validation_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.mean(torch.square(output - target))
        return loss

#I did this to allow faster computation of validation PSNR
class mse_loss_sum_across_batch():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.square(output - target))/(output.shape[1]*output.shape[2]*output.shape[3])
        return output.shape[0], loss
