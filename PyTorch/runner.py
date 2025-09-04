import logging
import os
import sys
import pickle
from torchinfo import summary
sys.path.append(os.getcwd())
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torchmetrics.image import SpectralAngleMapper
from utilities.utils import colorize_ms_mosaic_to_hw3, colorize_multichannel_image_with_triplets, get_device, get_model, delete_old_wandb_files, get_args, save_output_images_rgb, set_job_id, output_rescaling_quantization, setup_wandb_config, save_checkpoint
from utilities.multi_spectral_dataset import MSpectralDataset
from utilities.loss_func import mse_loss,psnr_per_image, psnr_per_image_masked, mse_loss_sum_across_batch, ssim_per_image
from PyTorch.utilities.color_space_converter import ColorSpaceConverter



def train_net(net,
              model_name,
              device,
              input_type,
              epochs=10,
              batch_size=32,
              lr=0.0001,
              lrdf=0.5,
              lrdp=25,
              chkpointperiod=1,
              trimages=0,
              patchsz=144,
              patchnum=4,
              validationFrequency=4,
              dir_img='../Scenes_npy_split/',
              num_workers = 4,
              save_cp=True,
              model = "",
              run_name = "",
              dir_checkpoint = None,
              job_id=None,
              odir = None,):

 
    if dir_checkpoint is None:
        dir_checkpoint = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../models/{run_name}/')

 
    train_dataset = MSpectralDataset(img_dir = dir_img, split = 'train', input_type = input_type, max_trdata=trimages)
    val_dataset = MSpectralDataset(img_dir = dir_img, split = 'val', input_type = input_type, max_trdata=0)
    
    
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    drop_last = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=drop_last)#setting batch size smaller to fit on single gpu

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Patches/image:   {patchnum}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device}
    ''')

    #if hsstaged optimizer should only be on hs_direct
    if model in ["fullpipeline"]:
        #turn off requires grad for the ms_backbone and rgb_backbone
        no_requires_grad_params = list(net.module.flow.parameters()) + list(net.module.ms_backbone.parameters()) + list(net.module.rgb_backbone.parameters()) 

        no_requires_grad_ids = {id(p) for p in no_requires_grad_params}

        params = [p for p in net.parameters() if id(p) not in no_requires_grad_ids]
         
        optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        for param in no_requires_grad_params:
            param.requires_grad = False
    else:
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lrdp, gamma=lrdf, last_epoch=-1)


    job_info_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../jobs/{args.job_id}.pkl')
    #load in job file details for vector jobs
    if os.path.exists(job_info_path):
        with open(job_info_path, 'rb') as fp:
            job_info_loaded = pickle.load(fp)
            args.load = job_info_loaded["last_checkpoint"]
            best_psnr = job_info_loaded["best_psnr"]
            if (torch.isinf(best_psnr)):
                best_psnr = 0
                print("LOADED INF PSNR")
            epochs_done = job_info_loaded["epochs_done"]
    else:
        best_psnr = 0
        epochs_done = 0
        
    
    if args.load:
        checkpoint = torch.load(args.load, map_location=device)

        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logging.info(f'Model loaded from {args.load}')
    

    #iter = 0
    for epoch in range(epochs_done, epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train*patchnum, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs_ = batch['in']
                gt_ = batch['out']
            
                assert imgs_.shape[1] == net.module.in_channels * patchnum, \
                    f'Network has been defined with {net.module.in_channels} input channels, ' \
                    f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'
                if model not in ["nafmspretrainrgb"]:
                    assert gt_.shape[1] == net.module.out_channels * patchnum, \
                    f'Network has been defined with {net.module.out_channels} output channels, ' \
                    f'but loaded GT images have {gt_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

                for j in range(patchnum):
                    imgs = imgs_[:, (j * net.module.in_channels): net.module.in_channels + (j * net.module.in_channels), :, :]
                    gt = gt_[:, (j * net.module.out_channels): net.module.out_channels + (j * net.module.out_channels), :, :]
                    
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    net_out = net(imgs)
                    
                    gt = gt.to(device=device, dtype=torch.float32)
                    
                    
                    if len(net_out) == 7:
                        imgs_pred, ms_out, rgb_from_ms, rgb_out, warped_rgb, flow, mask_invalid = net_out
                    elif len(net_out) == 4:
                        illumination, reflectance, reconstructed_spectra, imgs_pred = net_out
                    elif len(net_out) == 2 and model == "c2":
                        rgb_features, ms_features = net_out
                    elif len(net_out) == 2:
                        imgs_pred, _ = net_out
                    elif len(net_out) == 1:
                        imgs_pred, = net_out
                    else:
                        bilinear_in, bilateral_sigmas, imgs_pred = net_out

                    
                    if model in ["fullpipeline", "fullpipelinedouble"]:
                        loss = mse_loss.compute(imgs_pred, gt)
                    elif model in ["nafmspretrainrgb"]:
                        #use first 3 channels of imgs_pred and compare to gt
                        loss = mse_loss.compute(imgs_pred[:,0:3], gt)
                    else: 
                        loss = mse_loss.compute(imgs_pred, gt)
                    
                    epoch_loss += loss.item()
                    

                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.update(batch_size) #batch_size patches seen
                    global_step += 1
                    


                    #basis_plot = plot_lines_from_tensor(net.module.spectral_bank.clone())
                    log_dict = {'Loss/train': loss.item(), "Global Step": global_step, "Epoch": epoch}

                    if model in ["fullpipeline"]:
                        log_dict["Total Loss"] = loss.item()
                        

                    wandb.log(log_dict)
                    optimizer.zero_grad()

                    
        #validation
        if (epoch + 1) % validationFrequency == 0 and model not in ["c2"]:
            
            outputs= vald_net(net, model_name, val_loader, device, out_imgs_dir=odir, crop_outer_pixels=76, epoch=epoch, discriminator=discriminator, plot_images=False)
            wandb.log(outputs)
            
            val_psnr = outputs["PSNR Val"].cpu() #prevents a bug from checkpoint loading
            if(val_psnr > best_psnr):
                save_checkpoint(f'best.pth', dir_checkpoint, net, optimizer, scheduler, None, None, None)
                best_psnr = val_psnr
                logging.info(f'Saved new best model at PSNR: {best_psnr}!')

        scheduler.step()

        #checkpoint saving
        if save_cp and (epoch + 1) % chkpointperiod == 0:
            save_checkpoint(f'epoch{epoch + 1}.pth', dir_checkpoint, net, optimizer, scheduler, None, None, None)
            logging.info(f'Checkpoint {epoch + 1} saved!')
            #keep only 2 checkpoints at a time
            last_checkpoint_num = epoch + 1 - 2*chkpointperiod
            previous_checkpoint = dir_checkpoint + f'epoch{last_checkpoint_num}.pth'

            if (os.path.exists(previous_checkpoint)):
                os.remove(previous_checkpoint)
                logging.info(f'Checkpoint {last_checkpoint_num} deleted!')

        
        #do my job tracking
        if job_id is not None:
            job_info = {"last_checkpoint":dir_checkpoint + f'epoch{epoch + 1}.pth',
                                "wandb_run_id": wandb.run.id,
                                "best_psnr": best_psnr,
                                "epochs_done": epoch+1}
            if not os.path.exists(os.path.dirname(job_info_path)):
                os.makedirs(os.path.dirname(job_info_path), exist_ok=True)
                logging.info('Created Job Info directory')
            with open(job_info_path, 'wb') as fp:
                pickle.dump(job_info, fp)
                print(f'Saved Job Info at Epoch {epoch+1}')
    
    save_checkpoint(f'net.pth', dir_checkpoint, net, optimizer, scheduler, None, None, None)

    logging.info('Saved trained model!')
    logging.info('End of training')


def vald_net(net, name, loader, device, plot_images=False, out_imgs_dir=None,crop_outer_pixels = 120, epoch = 0, discriminator=None, csc=None, name_model_imgs="", out_special=None):
    """Evaluation using MAE"""
    net.eval()

    outputs = {}

    def log(key, value):
        if (key in outputs):
            outputs[key] += value
        else:
            outputs[key] = value
            
    #make psnrs an empty np array
    psnrs = np.array([])
    mask_invalid = None

    sam_metric = SpectralAngleMapper().to(device)
    with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
        count = 0
        for batch in loader:
            imgs_ = batch['in']
            gt_ = batch['out']
            patchnum = 1
            assert imgs_.shape[1] == net.module.in_channels * patchnum, \
                    f'Network has been defined with {net.module.in_channels} input channels, ' \
                    f'but loaded training images have {imgs_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'
            if name not in ["nafmspretrainrgb"]:
                assert gt_.shape[1] == net.module.out_channels * patchnum, \
                    f'Network has been defined with {net.module.out_channels} input channels, ' \
                    f'but loaded GT images have {gt_.shape[1] / patchnum} channels. Please check that ' \
                    'the images are loaded correctly.'

            imgs = imgs_[:, :, :, :]
            gt = gt_[:, :, :, :]
            imgs = imgs.to(device=device, dtype=torch.float32)

            gt = gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                if name in ["nafspec"]:
                    net_out = net(imgs, test=True)
                else:
                    net_out = net(imgs)
                
                if out_special in ["rgbgt", "gt"]:
                    imgs_pred = gt
                elif out_special in ["noise", "mosaic", "mosaic_sr"]:
                    ms_noise = batch['ms_noise'].to(device=device, dtype=torch.float32)
                    imgs_pred = ms_noise
                elif len(net_out) == 7:
                    imgs_pred, ms_out, rgb_from_ms, rgb_out, warped_rgb, flow, mask_invalid = net_out
                elif len(net_out) == 2:
                    imgs_pred, _ = net_out
                elif name in ["nafmspretrainrgb"]:
                    imgs_pred, = net_out
                    #get first 3 channels of imgs_pred
                    imgs_pred = imgs_pred[:,0:3]
                    
                elif len(net_out) == 1:
                    imgs_pred = net_out[0]

                imgs_pred = output_rescaling_quantization(imgs_pred)
                gt = output_rescaling_quantization(gt)
                        
                #batch size may be different for last batch
                batch_size, loss = mse_loss_sum_across_batch.compute(imgs_pred, gt)

                _, psnr_batch, psnrs_per_image = psnr_per_image.compute(imgs_pred, gt)

                
                print("PSNR Val", psnr_batch)
                if mask_invalid is not None:
                    _, psnr_masked_batch, psnrs_masked_per_image = psnr_per_image_masked.compute(imgs_pred, gt, ~mask_invalid)
                    log("PSNR Masked Val", psnr_masked_batch)

                    
                log("PSNR Val", psnr_batch)

                log("MSE Val", loss)
                log("Number of items", batch_size)
                
                #add psnrs_per_image to psnrs
                psnrs = np.concatenate((psnrs, psnrs_per_image.cpu().numpy()))
                
                #if the image size is bigger than 256x256, compute ssim
                if imgs.size(2) > 256:

                    _, ssim_batch, ssims_per_image = ssim_per_image.compute(imgs_pred, gt)
                    log("SSIM Val", ssim_batch)

                    #compute SAM
                    #small epsilon to prevent division by zero
                    epsilon = 1e-8
                    sam = sam_metric(imgs_pred + epsilon, gt + epsilon) * (180/3.14159) #this works only when the batch size is 1....
                    log("SAM Val", sam)
                    print("SAM Val", sam)

                    
                    #compute the PSNR per channel
                    psnrs_per_channel = []
                    for i in range(imgs_pred.size(1)):
                        _, psnr_batch_channel, psnrs_per_image_channel = psnr_per_image.compute(imgs_pred[:,i:i+1], gt[:,i:i+1])
                        psnrs_per_channel.append(psnrs_per_image_channel)
                        log(f"PSNR Val Channel {i}", psnr_batch_channel)
                        
            
                if plot_images:
                    
                    #if imgs_pred is 3 channels
                    ms_input = imgs_pred.size(1) == 16

                    rgb = csc(imgs_pred, ms_input=ms_input)
                    if len(net_out) == 7:
                        rgbs_warped = csc(torch.clamp(warped_rgb+0.5, 0, 1), ms_input=False)
                    else:
                        rgbs_warped = None
                    log_dict={}
                    
                    ms_colors = [
                        (0, 80, 20),
                        (255, 223, 0),
                        (255, 190, 0),
                        (255, 0, 0),
                        (106, 0, 255),
                        (0, 70, 255),
                        (0, 192, 200),
                        (0, 255, 0),
                        (94, 170, 0),
                        (240, 255, 0),
                        (60, 0, 0),
                        (106, 0, 255),
                        (0, 40, 255),
                        (0, 192, 255),
                        (0, 255, 146),
                        (40, 150, 40)
                    ]
                    
                    ms_scales = [8,2,4,2,4,4,10,4,4,4,4,7,4,4,4,10]
                    
                    if out_special in ["mosaic"]:
                        #make rgb the mosaic
                        ms_colored = colorize_multichannel_image_with_triplets(gt, ms_colors, ms_scales) #this works only when the batch size is 1....
                        rgb[0] = colorize_ms_mosaic_to_hw3(ms_colored)
                    elif out_special in ["mosaic_sr"]:
                        #downsample the gt with nearest neighbor
                        gt_downsampled = F.interpolate(gt, scale_factor=0.25, mode='nearest')
                        ms_colored_downsampled = colorize_multichannel_image_with_triplets(gt_downsampled[0], ms_colors, ms_scales) #this works only when the batch size is 1....
                        rgb_downsampled = colorize_ms_mosaic_to_hw3(ms_colored_downsampled)
                        #upsample the rgb_downsampled
                        rgb[0] = F.interpolate(rgb_downsampled.unsqueeze(0), scale_factor=4, mode='nearest')[0]
                            
                                  
                    #create folder in out_imgs_dir to contain npys
                    npys_path = os.path.join(out_imgs_dir, f'npys')
                    os.makedirs(npys_path, exist_ok=True)
                    #save the npys for Raw RGB, Intermediate MS, IntermediateMSConvRGB, Final MS
                    prefix = f"model_{name_model_imgs}_scene_{batch['scene_number'][0]}_position_{batch['position_number'][0]}_"

                    print("Name, ", name)
                    if name == "fullpipeline":

                        np.save(os.path.join(npys_path, f'{prefix}_rgb.npy'), output_rescaling_quantization(rgb_out[0]).cpu().numpy())
                        np.save(os.path.join(npys_path, f'{prefix}_rgbfromms.npy'), output_rescaling_quantization(rgb_from_ms[0]).cpu().numpy())
                        np.save(os.path.join(npys_path, f'{prefix}_intermediatems.npy'), output_rescaling_quantization(ms_out[0]).cpu().numpy())
                        np.save(os.path.join(npys_path, f'{prefix}_finalms.npy'), imgs_pred[0].cpu().numpy()) #already renormalized
                        np.save(os.path.join(npys_path, f'{prefix}_flow.npy'), flow[0].cpu().numpy())
                        np.save(os.path.join(npys_path, f'{prefix}_warped_rgb.npy'), warped_rgb[0].cpu().numpy())
                    else:
                        #just save the final ms for other models
                        np.save(os.path.join(npys_path, f'{prefix}_finalms.npy'), imgs_pred[0].cpu().numpy()) #already renormalized
                    
                    #save_output_images(imgs_pred, gt, rgb, rgbs_warped, psnrs_per_image, ssims_per_image, psnrs_per_channel, out_imgs_dir, name_model_imgs, batch['scene_number'], batch['position_number'], batch['view_number'])
                    save_output_images_rgb(rgb, out_imgs_dir, psnrs_per_image, ssims_per_image, name_model_imgs, batch['scene_number'], batch['position_number'], batch['view_number'])
                    wandb.log(log_dict)
                    
            pbar.update(1)

    for key in outputs.keys():
        if key != "Number of items":
            outputs[key] = outputs[key]/outputs["Number of items"]

    #remove the number of items key
    outputs.pop("Number of items")

    #set network back to train mode
    net.train()

    return outputs


if __name__ == '__main__':
    #set random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    args = set_job_id(args)
    gpus_chosen, device = get_device(args)
    net, discriminator, teacher = get_model(args)

    logging.info(f'Using model {args.model}')
    net.to(device)

    macs = 0
    params = 0


    net = torch.nn.DataParallel(net, device_ids=gpus_chosen)


    if discriminator is not None:
        discriminator = discriminator.to(device)
        discriminator = torch.nn.DataParallel(discriminator, device_ids=gpus_chosen)

    run_name = setup_wandb_config(args, net, macs, params)
    
    if not args.test:

        delete_old_wandb_files()

        #restart wandb progress if stuff got stopped in the middle
        job_info_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../jobs/{args.job_id}.pkl')
        if (os.path.exists(job_info_path)):
            resume="must"
            with open(job_info_path, 'rb') as fp:
                job_info_loaded = pickle.load(fp)
                run_name = job_info_loaded["wandb_run_id"]
        else:
            resume = False

        wandb.init(id = run_name,
        entity = 'cvil-yorku',
        project="SpectralDemosaicTrain",
        config = wandb.config,
        settings=wandb.Settings(code_dir="/home/tedlasai/spectral_demosaic/PyTorch"),
        name = run_name,
        notes="",
        resume=resume)

        odir = os.path.join(args.odir, run_name)

        logging.info('Training of Demosaicing Algorithms')

        train_net(net=net,
                  model_name = args.model,
                  input_type = args.input_type,
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    lr=args.lr,
                    lrdf=args.lrdf,
                    lrdp=args.lrdp,
                    device=device,
                    chkpointperiod=args.chkpointperiod,
                    trimages=args.trimages,
                    validationFrequency=args.val_frq,
                    patchsz=args.patchsz,
                    patchnum=args.patchnum,
                    dir_img=args.trdir,
                    num_workers=args.num_workers,
                    model = args.model,
                    run_name = run_name,
                    job_id = args.job_id,
                    odir=odir,)
                    
    else:
        wandb.init(project="SpectralDemosaicTest",
        entity = 'cvil-yorku',
        config = wandb.config,
        name = run_name,
        settings=wandb.Settings(code_dir="/home/tedlasai/spectral_demosaic/PyTorch"),
        notes="")
        
        if args.plot_images:
            odir = os.path.join(args.odir, run_name)
            os.makedirs(odir, exist_ok=True)
        else:
            odir = None
        logging.info('Testing of MS Demosaicing Algorithms')
        assert args.load, "specify the model to load" 

        checkpoint = torch.load(args.load, map_location=device)
        net.load_state_dict(checkpoint['net_state_dict'])
        
        num_channels = 1
        if args.model in ["fullpipeline"]:
            num_channels = 2

        height = 1440
        width = 2160
        if "ds4" in args.input_type and "fullpipeline" not in args.model:
            height = height//4
            width = width//4

        with torch.no_grad():
            model_stats = summary(net, input_size=(1, num_channels, height, width))
            macs = model_stats.total_mult_adds/(1.0e9)
            wandb.log({"MACs": macs})
        
        csc = ColorSpaceConverter()
        #make into data parallel
        csc = torch.nn.DataParallel(csc, device_ids=gpus_chosen)
        
        if args.plot_images:
            split = "val_test"
        else:
            split = "test"

        test_full_dataset = MSpectralDataset(args.trdir, split=split, input_type = args.input_type, full_size=True)
        test_full_loader = DataLoader(test_full_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
        if args.model in ["fullpipeline"]:
            modelname = args.backbone + "FP" 
        else:
            modelname = args.model
        
        if args.sr:
            modelname += "SR"

        out_special = args.out_special
        if out_special is not None:
            modelname = out_special
            odir = os.path.join(args.odir, modelname)
            os.makedirs(odir, exist_ok=True)
        
        
        outputs = vald_net(net, args.model, test_full_loader, device, out_imgs_dir=odir,plot_images=args.plot_images, crop_outer_pixels = 2, csc=csc, name_model_imgs=modelname, out_special =out_special)

        #add "Full" to all the keys
        for key in list(outputs.keys()):
            outputs[f"Full {key}"] = outputs.pop(key)
        
        wandb.log(outputs)