import torch
import torch.nn.functional as F

def load_pretrained_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')["net_state_dict"]
    pretrained_dict = {key.replace('module.', '', 1): value for key, value in checkpoint.items()}
    model.load_state_dict(pretrained_dict)
    return model     



def warp_image(image, flow, border_handle='constant', constant_value=-5):
    """
    Warp an image using the given flow map.

    :param image: The image to be warped. Tensor of shape (N, C, H, W)
    :param flow: The flow map used to warp the image. Tensor of shape (N, 2, H, W)
    :return: The warped image. Tensor of shape (N, C, H, W)
    """

    N, C, H, W = image.shape

    # Create a mesh grid of pixel coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # Shape (H, W, 2)
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # Shape (N, H, W, 2)

    # Normalize grid to the range [-1, 1]
    grid = grid.to(image.device)
    grid[..., 0] = (grid[..., 0] / (W)) * 2.0 - 1.0
    grid[..., 1] = (grid[..., 1] / (H)) * 2.0 - 1.0

    # Adjust grid with the flow
    flow = flow.permute(0, 2, 3, 1)  # Shape (N, H, W, 2)
    warped_grid = grid + flow

    if border_handle == 'constant':
        padding_mode = 'zeros'
    else:
        padding_mode = border_handle
    # Sample the image with the warped grid
    warped_image = F.grid_sample(image, warped_grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)
    
    if border_handle == 'constant':
        pixel_out_of_image = torch.logical_or(
            torch.logical_or(warped_grid[..., 0] < -1, warped_grid[..., 0] > 1),
            torch.logical_or(warped_grid[..., 1] < -1, warped_grid[..., 1] > 1)
        )
        #extend this mask to all channels
        pixel_out_of_image_repeated = pixel_out_of_image.unsqueeze(1).repeat(1, C, 1, 1)
        warped_image[pixel_out_of_image_repeated] = constant_value

    return pixel_out_of_image.unsqueeze(1), warped_image

def volume_of_homographies(image, flow, start_translation=1, end_translation=200, num_translations=20):
    """
    Creates a volume of homographies by translating the image along the median flow direction.
    
    :param image: The input image. Tensor of shape (N, C, H, W)
    :param flow: The flow map used to compute the median direction. Tensor of shape (N, 2, H, W)
    :param max_translation: The maximum translation distance in pixels.
    :param step: The step size for translation in pixels.
    :return: A tuple containing:
        - warped_images: Tensor of warped images with shape (N, num_translations, C, H, W)
        - invalid_masks: Tensor of invalid masks with shape (N, num_translations, 1, H, W)
    """
    
    N, C, H, W = image.shape
    
    # Step 1: Compute the median direction of the flow (median across H, W dimensions)
    median_flow = torch.median(flow.view(N, 2, -1), dim=-1)[0]  # Shape (N, 2)
    
    # Normalize the flow direction (median_flow will serve as the direction)
    direction = median_flow / (torch.norm(median_flow, dim=1, keepdim=True) + 1e-8)  # Shape (N, 2)
    
    # Step 2: Generate translations in the median flow direction
    translations = torch.linspace(start_translation, end_translation, num_translations, device=image.device).view(1, num_translations, 1, 1, 1)  # Shape (1, num_translations, 1, 1)
    
    # Step 3: Create translation flow for all images in the median direction
    translation_flows = direction.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) * translations  # Shape (N, num_translations, 2, 1, 1)
    translation_flows = translation_flows.expand(N, num_translations, 2, H, W)  # Shape (N, num_translations, 2, H, W)
    
    translation_flows = translation_flows.clone()
    
    # Step 4: Normalize translation flows to the range [-1, 1]
    translation_flows[..., 0, :, :] /= W  # Normalize the x-component (horizontal) by image width
    translation_flows[..., 1, :, :] /= H  # Normalize the y-component (vertical) by image height
    translation_flows = translation_flows * 2.0  # Since warp_image expects values between -1 and 1
    
    # Step 5: Warp the image for all translations in a single call
    image_expanded = image.unsqueeze(1).expand(N, num_translations, C, H, W).reshape(-1, C, H, W)  # Shape (N * num_translations, C, H, W)
    translation_flows = translation_flows.reshape(-1, 2, H, W)  # Shape (N * num_translations, 2, H, W)
    
    # Step 6: Warp the images using the expanded translation flows and get the invalid masks
    invalid_masks, warped_images = warp_image(image_expanded, translation_flows)
    
    # Step 7: Reshape the output back to (N, num_translations, C, H, W)
    warped_images = warped_images.view(N, num_translations, C, H, W)
    
    # Step 8: Reshape invalid masks back to (N, num_translations, 1, H, W)
    invalid_masks = invalid_masks.view(N, num_translations, 1, H, W)
    
    return invalid_masks, warped_images



def warp_image_with_offset(image, flow_map, patch_size=64, offset=32):
    """
    Warp an image using flow maps with given patch size and offset.

    Args:
        image: Tensor of shape N*C*H*W (input image to warp).
        flow_map: Tensor of shape N*2*H*W (flow in x and y directions).
        patch_size: Size of the square patches.
        offset: Offset to handle overlapping patches.

    Returns:
        warped_image_offset: Tensor of shape N*C*H*W (warped image with patches starting at offset).
    """

    N, C, H, W = image.shape
    
    # Compute flow for patches starting at offset
    avg_flow_offset = torch.nn.functional.avg_pool2d(flow_map[:, :, offset:, offset:], kernel_size=patch_size, stride=patch_size)
    avg_flow_offset = torch.nn.functional.pad(avg_flow_offset, (offset, offset, offset, offset), mode='constant', value=0)
    avg_flow_offset_upsampled = torch.nn.functional.interpolate(avg_flow_offset, size=(H, W), mode='nearest')
    

    # Create the base grid for the entire image (normalized to [-1, 1])
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H),  # Corresponds to normalized y coordinates
        torch.linspace(-1, 1, W)   # Corresponds to normalized x coordinates
    )

    # Stack the grid to get a (H, W, 2) shape
    base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).to(image.device)

    # Apply the flow to the grid for patches starting at offset
    warped_grid = base_grid + avg_flow_offset_upsampled.permute(0, 2, 3, 1)

    # Warp the entire image using the computed grid
    warped_image_offset = torch.nn.functional.grid_sample(image, warped_grid, mode='nearest', align_corners=True)
    
        #print MSE beeteween warped_image_offset and image

    
    pixel_out_of_image = torch.logical_or(
    torch.logical_or(warped_grid[..., 0] < -1, warped_grid[..., 0] > 1),
    torch.logical_or(warped_grid[..., 1] < -1, warped_grid[..., 1] > 1)
    )
    #extend this mask to all channels
    pixel_out_of_image_repeated = pixel_out_of_image.unsqueeze(1).repeat(1, C, 1, 1)
    #if a pixel is out of the image, set it to -5
    warped_image_offset[pixel_out_of_image_repeated] = -5
    
    
    

    return pixel_out_of_image.unsqueeze(1), warped_image_offset
