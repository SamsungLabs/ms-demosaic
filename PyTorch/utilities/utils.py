from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import glob
from shutil import rmtree
import time
from pathlib import Path
import logging
from PIL import Image
from datetime import datetime
import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from types import SimpleNamespace
import yaml

from simple_camera_pipeline.python.pipeline import run_pipeline_v2
from simple_camera_pipeline.python.pipeline_utils import get_metadata
sys.path.append("..")
import wandb
from PyTorch.arch import full_pipeline_model, mcan_model, nafnet_ind, restormer_model
import torchvision

def export_image(filename, img):
    np.save(filename, img)


def build_pattern(order = "RGGB", filter_size = 1, rows = 300, cols = 300, pattern_dict = {"R":0, "G": 1, "B": 2}):
    pattern_tile = torch.empty((filter_size*2, filter_size*2), dtype=torch.int64)
    for row in range(2):
        for col in range(2):
            pattern_index = row*2 + col
            channel_letter = order[pattern_index]
            pattern_channel = pattern_dict[channel_letter]
            pattern_tile[row*filter_size:(row+1)*filter_size, col*filter_size:(col+1)*filter_size] = pattern_channel
    pattern = torch.tile(pattern_tile, (rows//(filter_size*2), cols//(filter_size*2)))
    return pattern


def construct_mosaic(image, order = "RGGB", filter_size = 1):
    pattern = build_pattern(order=order, filter_size=filter_size, rows=image.shape[0], cols=image.shape[1])
    #x = torch.arange(0, image.shape[1])
    #y = torch.arange(0, image.shape[0])
    #col_mesh, row_mesh = torch.meshgrid(x,y, indexing='ij') #meshgrid is wierd(but this is how you do it)
    out = torch.gather(input=image, dim=2, index=pattern.unsqueeze_(dim=-1)).squeeze()
    return pattern, out




def calc_model_size(model):
    # From: https://discuss.pytorch.org/t/finding-model-size/130275
    # ~~ in @ptrblck we trust ~~
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()


    size_all_mb = (param_size + buffer_size) / (1024 ** 2) # for MB

    return size_all_mb

#move this to a utils file
def get_freer_gpu(num_gpus):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])

        #this means I am on the other server
        if len(memory_available) == 0:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
            memory_used = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
            return np.argsort(memory_used)[0:num_gpus]
        else:
            return np.argsort(-1*memory_available)[0:num_gpus]
    
#from deep_architect with MIT License
def get_total_num_gpus():
    try:
        import subprocess
        n = len(subprocess.check_output(['nvidia-smi','-L']).decode('utf-8').strip().split('\n'))
    except OSError:
        n = 0
    return n 

#make this a function
    

def delete_old_wandb_files():
    for file_location in glob.glob(f'/tmp/*wandb*'):
        # file_time is the time when the file is modified
        file_time = os.stat(file_location).st_mtime

        # if a file is modified before N days then delete it
        N=7
        current_time = time.time()
        day = 86400 #seconds in a day

        path = Path(file_location)
        owner = path.owner()
        if (owner == "tedlasai"):
            if(file_time < (current_time - day*N)):
                print(f" Delete : {file_location}")
                rmtree(file_location)


def get_device(args):
    if args.num_gpus <= 4:
        gpus_chosen = get_freer_gpu(args.num_gpus).tolist()
        logging.info(f'Using devices {gpus_chosen}')

        all_gpus = [i for i in range(get_total_num_gpus())]
        all_gpus.sort()

        logging.info(f'ALL GPUS {all_gpus}')

        gpus_chosen = [0]

       # gpus_chosen=[0]

        #os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, all_gpus))
        os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] ="50"
        device = f"cuda:{gpus_chosen[0]}"
    else:
        device = "cuda:0"
        #create a list of length args.num_gpus
        gpus_chosen = [i for i in range(args.num_gpus)]
        os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, gpus_chosen))
        print("Gpus chosen", gpus_chosen)
    return gpus_chosen, device




def get_args():
    parser = argparse.ArgumentParser(description="YAML config loader")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    return SimpleNamespace(**config)

def get_args_old():
    parser = argparse.ArgumentParser(description='Train demosaicing network.', add_help=False)
    parser.add_argument('-g', '--num-gpus', type=int, default=1, dest='num_gpus',
                    help='Number of GPUs to use')


    # --- Training settings ---
    parser.add_argument('-e', '--epochs', type=int, default=80, dest='epochs',
                        help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=16, dest='batchsize',
                        help='Batch size')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001, dest='lr',
                        help='Learning rate')
    parser.add_argument('-lrdf', '--lr-decay-factor', type=float, default=0.5, dest='lrdf',
                        help='Learning rate decay factor')
    parser.add_argument('-lrdp', '--lr-decay-period', type=int, default=25, dest='lrdp',
                        help='Learning rate decay period')
    parser.add_argument('-c', '--checkpoint-period', type=int, default=1, dest='chkpointperiod',
                        help='Number of epochs to save a checkpoint')
    parser.add_argument('-t', '--num_training_images', type=int, default=0, dest='trimages',
                        help='Number of training images')
    parser.add_argument('-vf', '--validation-frequency', type=int, default=1, dest='val_frq',
                        help='Validation frequency')
    parser.add_argument('-p', '--patches-per-image', type=int, default=1, dest='patchnum',
                        help='Number of training patches per image')
    parser.add_argument('-s', '--patch-size', type=int, default=256, dest='patchsz',
                        help='Size of training patch')

    # --- Model & input settings ---
    parser.add_argument('-m', '--model', type=str, default="mfk", dest='model',
                        help='Model used')
    parser.add_argument('-it', '--input-type', type=str, default="msrgb_mosaic", dest='input_type',
                        help='Input type')
    parser.add_argument('-sr', '--super-resolution', default=False, action='store_true', dest='sr',
                        help='Super resolution problem')
    parser.add_argument('-os', '--out_special', default=None, dest='out_special',
                        help='Special Output')
    parser.add_argument('-bn', '--backbone-network', type=str, default="restormer", dest='backbone',
                        help='Backbone network for fullpipeline')

    # --- Paths & checkpointing ---
    parser.add_argument('-trd', '--training_dir', default='/local/ssd/tedlasai/ms', dest='trdir',
                        help='Training image directory')
    parser.add_argument('-od', '--odir', default="/datasets/sai/spectral_demosaic_network_outputs", dest='odir',
                        help='Output Directory')
    parser.add_argument('-l', '--load', type=str, default=False, dest='load',
                        help='Load model from a .pth file')
    parser.add_argument('-ji', '--job-id', type=int, default=None, dest='job_id',
                        help='Vector Job Id')

    # --- Dataloader ---
    parser.add_argument('-w', '--num-workers', type=int, default=16, dest='workers',
                        help='Number of dataloader workers')

    # --- Testing & visualization ---
    parser.add_argument('-test', '--test', default=False, action='store_true', dest='test',
                        help='Run in test mode')
    parser.add_argument('-pi', '--plot_images', default=False, action='store_true', dest='plot_images',
                        help='Output images')


    return parser.parse_args()

def set_job_id(args):
    if args.job_id is None:
        args.job_id = int(time.time())
    return args

def setup_wandb_config(args, net, macs, params):

    model_size = calc_model_size(net)

    #conbine args and model size into a dict
    wandb.config = {**vars(args), "model_size": model_size, "macs": macs, "params": params}

    currentSecond= datetime.now().second
    currentMinute = datetime.now().minute
    currentHour = datetime.now().hour
    currentDay = datetime.now().day
    currentMonth = datetime.now().month
    
    run_name = f"{args.model}_{args.backbone}_{args.input_type}_{currentMonth}-{currentDay}-{currentHour}-{currentMinute}-{currentSecond}"
    return run_name

    
def convert_image(img):
    metadata = get_metadata("/home/tedlasai/demosaic/DSC01201_PSMS.dng")

    params = {"input_stage": "demosaic", "output_stage": "tone"}
    img = run_pipeline_v2(img[:,:,:], params, metadata)
    #clip 
    img = np.clip(img[:,:,:], 0, 1)
    img = (img *255).astype(np.uint8)

    return img



def get_model(args):
    discriminator = None
    teacher = None

    if args.model == "mcan":
        net = mcan_model.MCANModel(sr=args.sr)
    elif args.model == "nafms":
        net = nafnet_ind.NafNetIndModel(in_channels=1, out_channels=16, sr=args.sr)
    elif args.model == "nafrgb":
        net = nafnet_ind.NafNetIndModel(in_channels=1, out_channels=3, size="small")
    elif args.model in ["fullpipeline", "fullpipelinedouble"]:
        net = full_pipeline_model.FullPipeline(args.backbone, sr=args.sr)
    elif args.model == "restormer":
        net = restormer_model.RestormerModel(in_channels=args.input_channels, out_channels=16, sr=args.sr)

        
    else:
        raise Exception ("Not valid model type")

    return net, discriminator, teacher

def output_rescaling_quantization(imgs):
    saved_device = imgs.device
    imgs = imgs+0.5
    imgs = imgs.clamp(0, 1)
    return imgs

def get_flow_color(flow):
    flow = flow.detach().cpu() #detach flow - this prevents memory leaks 
    #clamp flow
    flow = (flow + 1)/2
    # Convert flow to image using torchvision
    flow_color = torchvision.utils.flow_to_image(flow)

    return flow_color

def colorize_ms_mosaic_to_hw3(image):
    """
    Convert an H*W*16*3 image to an H*W*3 colorized image by assigning each 
    of the 16 channels to a specific position in a 4x4 pattern within each pixel.

    Parameters:
    - image: np.array of shape (H, W, 16, 3), the input multispectral mosaic image.

    Returns:
    - output_image: np.array of shape (H, W, 3), the final colorized mosaic image.
    """
    _, C, H, W = image.shape
    assert C == 16, "Expected 16 channels in the image."

    # Initialize the output image
    output_image = torch.zeros((3, H, W), dtype=image.dtype)


    s = 4
    for i in range(s):
        for j in range(s):
            output_image[:, i::s, j::s] = image[ :, i * s + j,i::s, j::s]

    return output_image

def colorize_multichannel_image_with_triplets(image, channel_colors, scales):
    """
    Convert an H*W*16 multichannel image to a single H*W*3 RGB image
    using specified RGB triplets for each channel.

    Parameters:
    - image: torch.Tensor of shape (H, W, C), the multichannel input image.
    - channel_colors: list of C tuples, 3 RGB color values for each of the C channels.
    - scales: list of C scale values for each channel
    - max: maximum color value (default: 255)

    Returns:
    - colored_image: torch.Tensor of shape (H, W, 3), the colorized output image.
    """
    C,H,W = image.shape

    # Initialize an empty RGB output image
    colored_image = torch.zeros((3, C, H, W), dtype=torch.float32).to(image.device)

    # Apply each RGB color triplet to its respective channel and add to the output image
    for i in range(C):
        r_color = channel_colors[i][0] / 255
        g_color = channel_colors[i][1] / 255
        b_color = channel_colors[i][2] / 255
        
        # Blend each channel color with the normalized channel data
        colored_image[0, i, :, :] += image[i, :, :] * r_color * scales[i]  # Red channel
        colored_image[1, i, :, :] += image[i, :, :] * g_color * scales[i]  # Green channel
        colored_image[2, i, :, :] += image[i, :, :] * b_color * scales[i]  # Blue channel

    # Clip values to [0, 1] to prevent overflow in displayable range
    colored_image = torch.clamp(colored_image, 0, 1)

    return colored_image


def save_output_images_rgb(rgbs: torch.Tensor, output_dir: str, psnrs, ssims, model_name:str, scene_nums: torch.Tensor, position_nums:torch.Tensor, view_nums: torch.Tensor):
    """
    Save RGB images and corresponding PSNR values with given scene and view numbers.
    Additionally, save each channel of the MS tensor as a turbo color-mapped image.
    
    Parameters:
    - ms (torch.Tensor): Input tensor of shape (B, 16, H, W) representing multi-spectral images.
    - gt (torch.Tensor): Ground truth tensor of shape (B, 16, H, W) representing ground truth multi-spectral images.
    - rgbs (torch.Tensor): Input tensor of shape (B, 3, H, W) representing RGB images.
    - rgbs_warped (torch.Tensor): Input tensor of shape (B, 3, H, W) representing RGB images warped using the predicted flow.
    - psnrs (torch.Tensor): Tensor of shape (B, 1) containing PSNR values for each image.
    - ssims (torch.Tensor): Tensor of shape (B, 1) containing SSIM values for each image.
    - psnrs_per_image_channel (list): List of length 16 containaing Tensors of shape (B,1) containing PSNR values for each channel of each image.
    - output_dir (str): Directory where output files will be saved.
    - model_name (str): Name assigned to the model.
    - scene_nums (torch.Tensor): Tensor of shape (B, 1) containing scene numbers for each image.
    - view_nums (torch.Tensor): Tensor of shape (B, 1) containing view numbers for each image.
    """
    B, C_rgb, H, W = rgbs.shape
    assert C_rgb == 3, "Input tensor must have 3 channels (RGB)."
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_out_dir = os.path.join(output_dir, 'rgb_out')

    # Create subdirectories if they don't exist
    os.makedirs(rgb_out_dir, exist_ok=True)

    # Loop through each batch item
    for i in range(B):
        # Save RGB image
        rgb_img = rgbs[i].cpu().numpy().transpose(1, 2, 0)  # Change from (3, H, W) to (H, W, 3)
        rgb_img = (rgb_img * 255).clip(0, 255).astype('uint8')  # Assuming the tensor is normalized between [0, 1]
        
        scene_num = scene_nums[i]
        view_num = view_nums[i]
        position_num = position_nums[i]
        psnr_value = psnrs[i].item()
        ssim_value = ssims[i].item()
        
        rgb_filename = f"model_{model_name}_scene_{scene_num}_position_{position_num}_view_{view_num}_psnr_{psnr_value:.2f}_ssim{ssim_value:.4f}.png"
        rgb_file_path = os.path.join(rgb_out_dir, rgb_filename)
        
        img_pil = Image.fromarray(rgb_img)
        img_pil.save(rgb_file_path)
        print(f"Saved RGB image: {rgb_file_path}")
        

def save_output_images(ms: torch.Tensor, gt: torch.Tensor, rgbs: torch.Tensor, rgbs_warped: torch.Tensor, psnrs: torch.Tensor, ssims: torch.Tensor, psnrs_per_channel: list, output_dir: str, model_name:dir, scene_nums: torch.Tensor, position_nums:torch.Tensor, view_nums: torch.Tensor):
    """
    Save RGB images and corresponding PSNR values with given scene and view numbers.
    Additionally, save each channel of the MS tensor as a turbo color-mapped image.
    
    Parameters:
    - ms (torch.Tensor): Input tensor of shape (B, 16, H, W) representing multi-spectral images.
    - gt (torch.Tensor): Ground truth tensor of shape (B, 16, H, W) representing ground truth multi-spectral images.
    - rgbs (torch.Tensor): Input tensor of shape (B, 3, H, W) representing RGB images.
    - rgbs_warped (torch.Tensor): Input tensor of shape (B, 3, H, W) representing RGB images warped using the predicted flow.
    - psnrs (torch.Tensor): Tensor of shape (B, 1) containing PSNR values for each image.
    - ssims (torch.Tensor): Tensor of shape (B, 1) containing SSIM values for each image.
    - psnrs_per_image_channel (list): List of length 16 containaing Tensors of shape (B,1) containing PSNR values for each channel of each image.
    - output_dir (str): Directory where output files will be saved.
    - model_name (str): Name assigned to the model.
    - scene_nums (torch.Tensor): Tensor of shape (B, 1) containing scene numbers for each image.
    - view_nums (torch.Tensor): Tensor of shape (B, 1) containing view numbers for each image.
    """
    B, C_rgb, H, W = rgbs.shape
    assert C_rgb == 3, "Input tensor must have 3 channels (RGB)."
    
    B_ms, C_ms, H_ms, W_ms = ms.shape
    assert C_ms == 16, "MS tensor must have 16 channels."
    assert gt.shape == ms.shape, "Ground truth (gt) and predicted (ms) tensors must have the same shape."
    assert H_ms == H and W_ms == W, "MS and RGB tensors must have the same spatial dimensions."

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_out_dir = os.path.join(output_dir, 'rgb_out')
    ms_out_dir = os.path.join(output_dir, 'ms_out')
    error_out_dir = os.path.join(output_dir, 'error_maps')

    # Create subdirectories if they don't exist
    os.makedirs(rgb_out_dir, exist_ok=True)
    os.makedirs(ms_out_dir, exist_ok=True)
    os.makedirs(error_out_dir, exist_ok=True)

    # Loop through each batch item
    for i in range(B):
        # Save RGB image
        rgb_img = rgbs[i].cpu().numpy().transpose(1, 2, 0)  # Change from (3, H, W) to (H, W, 3)
        rgb_img = (rgb_img * 255).clip(0, 255).astype('uint8')  # Assuming the tensor is normalized between [0, 1]
        
        scene_num = scene_nums[i]
        view_num = view_nums[i]
        position_num = position_nums[i]
        psnr_value = psnrs[i].item()
        ssim_value = ssims[i].item()
        
        rgb_filename = f"model_{model_name}_scene_{scene_num}_position_{position_num}_view_{view_num}_psnr_{psnr_value:.2f}_ssim{ssim_value:.4f}.png"
        rgb_file_path = os.path.join(rgb_out_dir, rgb_filename)
        
        img_pil = Image.fromarray(rgb_img)
        img_pil.save(rgb_file_path)
        print(f"Saved RGB image: {rgb_file_path}")
        
        if rgbs_warped is not None:
            # Save RGB warped image
            rgb_warped_img = rgbs_warped[i].cpu().numpy().transpose(1, 2, 0)  # Change from (3, H, W) to (H, W, 3)
            rgb_warped_img = (rgb_warped_img * 255).clip(0, 255).astype('uint8')  # Assuming the tensor is normalized between [0, 1]
            
            rgb_warped_filename = f"model_{model_name}_scene_{scene_num}_position_{position_num}_view_{view_num}_psnr_{psnr_value:.2f}_ssim{ssim_value:.4f}_warped.png"
            rgb_warped_file_path = os.path.join(rgb_out_dir, rgb_warped_filename)
            
            warped_img_pil = Image.fromarray(rgb_warped_img)
            warped_img_pil.save(rgb_warped_file_path)
            
            print(f"Saved RGB warped image: {rgb_warped_file_path}")

        max_error_map = 0.15
        ch_scale = {0:6, 1:1, 2:2, 3:1, 4:1,5:1, 6:8, 7:1, 8:1.5, 9:2, 10:1, 11:5, 12:1.5, 13:1.5, 14:1,15:10}
        # Save each channel of the MS image as a turbo color-mapped image
        for ch in range(16):
            ms_channel = ms[i, ch].cpu().numpy()  # Get the channel (H, W)
            # Generate and save per-channel error map
            #compute mse between ms and gt
            error_map = np.sqrt((ms_channel - gt[i, ch].cpu().numpy())**2)
            mse = format(np.mean(error_map), ".2e")
            
            #max color map is 0.5
            ms_channel = ms_channel*ch_scale[ch]
            ms_channel = np.clip(ms_channel, 0, 1)
            # Apply the turbo colormap
            turbo_colormap = cm.get_cmap('turbo')
            ms_colored = turbo_colormap(ms_channel)[:, :, :3]  # Get RGB from colormap (drop alpha channel)
            ms_colored = (ms_colored * 255).astype('uint8')  # Scale to [0, 255]

            psnr = psnrs_per_channel[ch][i].item()
            
            ms_filename = f"model_{model_name}_scene_{scene_num}_position_{position_num}_view_{view_num}_channel_{ch}_psnr_{psnr:.2f}.png"
            ms_file_path = os.path.join(ms_out_dir, ms_filename)
            
            # Save the color-mapped image
            ms_img_pil = Image.fromarray(ms_colored)
            ms_img_pil.save(ms_file_path)
            print(f"Saved MS channel {ch} image: {ms_file_path}")

            #max of error map is "0.1"
            
            error_map = np.clip(error_map/max_error_map, 0, 1)
            
            # Apply the hot colormap
            hot_colormap = cm.get_cmap('turbo')
            error_colored = hot_colormap(error_map)[:, :, :3]  # Get RGB from colormap (drop alpha channel)
            error_filename = f"model_{model_name}_scene_{scene_num}_position_{position_num}_view_{view_num}_channel_{ch}_mse_{mse}_error.png"
            error_file_path = os.path.join(error_out_dir, error_filename)
            
            # Save the error map
            error_img = (error_colored * 255).astype('uint8')  # Scale error map to [0, 255]
            error_img_pil = Image.fromarray(error_img)
            error_img_pil.save(error_file_path)
            print(f"Saved error map for channel {ch}: {error_file_path}")
        
        #sum all channels and visualize that
        sum_ms = np.mean(ms[i].cpu().numpy(), axis=0) * 2
        sum_ms_colored = turbo_colormap(sum_ms)[:, :, :3]  # Get RGB from colormap (drop alpha channel)
        sum_ms_colored = (sum_ms_colored * 255).astype('uint8')  # Scale to [0, 255]
        sum_ms_filename = f"model_{model_name}_scene_{scene_num}_position_{position_num}_view_{view_num}_sum.png"
        sum_ms_file_path = os.path.join(ms_out_dir, sum_ms_filename)
        
        # Save the color-mapped image
        sum_ms_img_pil = Image.fromarray(sum_ms_colored)
        sum_ms_img_pil.save(sum_ms_file_path)
        print(f"Saved sum of MS channels image: {sum_ms_file_path}")
            
        
        #Now do same heat map across all channels
        #compute mse between ms and gt
        error_map = np.sqrt((ms[i].cpu().numpy() - gt[i].cpu().numpy())**2)
        #sum across all channels
        error_map = np.mean(error_map, axis=0)
        mse = format(np.mean(error_map), ".2e")
        error_map = np.clip(error_map/max_error_map, 0, 1)
        # Apply the hot colormap
        hot_colormap = cm.get_cmap('turbo')
        error_colored = hot_colormap(error_map)[:, :, :3]  # Get RGB from colormap (drop alpha channel)
        error_filename = f"model_{model_name}_scene_{scene_num}_position_{position_num}_view_{view_num}_mse_{mse}_error.png"
        error_file_path = os.path.join(error_out_dir, error_filename)
        
        # Save the error map
        error_img = (error_colored * 255).astype('uint8')  # Scale error map to [0, 255]
        error_img_pil = Image.fromarray(error_img)
        error_img_pil.save(error_file_path)
        print(f"Saved error map for channel {ch}: {error_file_path}")
        
        


def generate_wandb_images(imgs, filenames, psnrs):
    out = []
    for i, image in enumerate(imgs[:2].cpu().detach().numpy()): #only do two patches
        image = image/13496 #renormalize
        image = np.clip(image, 0, 1)
        patch_gamma = np.moveaxis(image, [0,1,2], [2,0,1]) **(1/2.2)
        upscale_factor = 2
        patch_upscaled = patch_gamma.repeat(upscale_factor, 1).repeat(upscale_factor, 0)
    return out

def save_checkpoint(checkpoint_name, dir_checkpoint, net, optimizer, scheduler, discriminator, optimizer_discriminator, scheduler_discriminator):
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint, exist_ok=True)
        logging.info('Created checkpoint directory')
    state_dict = {
    'net_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    }

    if discriminator is not None:
        state_dict["discriminator_state_dict"] = discriminator.state_dict()
        state_dict["optimizer_discriminator_state_dict"] = optimizer_discriminator.state_dict()
        state_dict["scheduler_discriminator_state_dict"] = scheduler.state_dict()

    torch.save(state_dict, dir_checkpoint + f'{checkpoint_name}')
    

# def viz_images(imgs, imgs_pred, gt):
    