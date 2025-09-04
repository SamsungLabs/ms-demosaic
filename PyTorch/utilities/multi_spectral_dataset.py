import numpy as np
from torch.utils.data import Dataset
import logging
import re
import os

def get_all_npy_files(base_dir):
    npy_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files


def create_mosaic_from_ms_or_rgb(img, pattern, ms_mean=None):
    H, W, C = img.shape
    mosaic = np.zeros((H, W), dtype=img.dtype)

    if pattern == 'ms':
        # Assuming a 4x4 pattern for 16 channels
        s = 4
        for i in range(s):
            for j in range(s):
                mosaic[i::s, j::s] = img[i::s, j::s, i * s + j]

    elif pattern == 'rgb':
        # RGGB pattern
        s = 2
        for i in range(s):
            for j in range(s):
                idx = s * i + j
                if(idx == 0):
                    mosaic[i::s, j::s] = img[i::s, j::s, 0]
                elif (idx == 1 or idx == 2):
                    mosaic[i::s, j::s] = img[i::s, j::s, 1]
                else:
                    mosaic[i::s, j::s] = img[i::s, j::s, 2]
    else:
        raise ValueError("Unknown mosaic pattern")
    mosaic_mean = None
    return mosaic, mosaic_mean, img

class MSpectralDataset(Dataset):
    def __init__(self, img_dir, split, input_type, dropout=0, patch_size=256, max_trdata=0, full_size=False):
        self.img_dir = img_dir
        self.split = split
        self.input_type = input_type
        self.dropout = dropout
        self.patch_size = patch_size
        self.full_size = full_size

        # Validate split and input type
        assert split in ['train', 'val', 'test', 'val_test'], "Unknown split"
        assert input_type in ['ms_mosaic', 'ms_mosaic_ds4', 'rgb_mosaic_ds4', 'rgb_mosaic', 'rgb_mosaic_view_2', 'msrgb_mosaic', 'rgb_mosaic_predict_ms', 'rgb_mosaic_inline_w_ms', 'ms_mosaic_inline', 'rgb_mosaic_w_fullms_inline', 'ms_mosaic_inline_w_rgb', 'ms_mosaic_ds4_inline_w_rgb', 'teacher_ms', 'rgb_mosaic_predicting_rgb', 'ms_mosaic_predicting_ms_and_rgb'], "Unknown input type"

        logging.info('Loading images information...')

        
        #walk through the directory and get all the files
        if full_size:
            self.ms_files = get_all_npy_files(os.path.join(img_dir, "ms_npys/iso400"))
            self.rgb_files = get_all_npy_files(os.path.join(img_dir, "rgb_npys/iso400"))
        else:
            self.ms_files = get_all_npy_files(os.path.join(img_dir, "cropped_ms_npys/iso400"))
            self.rgb_files = get_all_npy_files(os.path.join(img_dir, "cropped_rgb_npys/iso400"))

        # Filter files based on the split
        if split == 'train':
            scene_indices = ['01','03', '04','05', '06','07','08','09','10','11','12','14','15', '16', '17', '18', '19', '20', '21', '25'] 
        elif split == 'val':
            scene_indices = ['23', '24'] 
        elif split == 'test':
            scene_indices = ['02', '13', '22', '26', '27', '28']
        elif split == "val_test":
            scene_indices = ['02', '13', '22', '23', '24', '26', '27', '28']
            
            
        self.ms_files = sorted([f for f in self.ms_files if any(f'scene{index}' in f for index in scene_indices)])
        self.rgb_files = sorted([f for f in self.rgb_files if any(f'scene{index}' in f for index in scene_indices)])
        
        #if file contains view02 or view03, remove it
        self.ms_files = [f for f in self.ms_files if ('view02' not in f and 'view03' not in f)]
        self.rgb_files = [f for f in self.rgb_files if ('view01' not in f and 'view03' not in f)]
        
        print("MS Files: ", len(self.ms_files))
        print("RGB Files: ", len(self.rgb_files))
        print("Length of MS Files: ", len(self.ms_files))
        print("Length of RGB Files: ", len(self.rgb_files))
        assert len(self.ms_files) == len(self.rgb_files), "Number of multi-spectral and RGB files do not match"

  
        if max_trdata != 0 and len(self.ms_files) > max_trdata:
            self.ms_files = self.ms_files[0:max_trdata]
            self.rgb_files = self.rgb_files[0:max_trdata]

        logging.info(f'Creating {split} dataset with {len(self.ms_files)} examples')

    def __len__(self):
        return len(self.ms_files)

    def preprocess(self, ms_img, rgb_img, ms_mean=None, rgb_mean=None, ds_factor=1, ds_rgb=False):
        
        if ds_factor != 1:
            H, W, C = ms_img.shape
            downsampled_img = lambda ms_img, factor: ms_img.reshape(H//factor, factor, W//factor, factor, 16).mean((1, 3))
            ms_img = downsampled_img(ms_img, ds_factor)
            
            if ds_rgb:
                H, W, C = rgb_img.shape
                downsampled_img = lambda rgb_img, factor: rgb_img.reshape(H//factor, factor, W//factor, factor, 3).mean((1, 3))
                rgb_img = downsampled_img(rgb_img, ds_factor)

        ms_mosaic, ms_mean, ms_demosaiced = create_mosaic_from_ms_or_rgb(ms_img, "ms", ms_mean)
        ms_demosaiced = ms_demosaiced -0.5 # 0 -1 -> -0.5 to 0.5
        ms_demosaiced = np.transpose(ms_demosaiced, (2,0,1)).astype(np.float32)
        ms_mosaic =  ms_mosaic - 0.5 # 0 -1 -> -0.5 to 0.5
        ms_mosaic = ms_mosaic.astype(np.float32)[np.newaxis, ...] #C*H*W


        rgb_mosaic, rgb_mean, rgb_demosaiced = create_mosaic_from_ms_or_rgb(rgb_img, "rgb", rgb_mean)
        rgb_mosaic =  rgb_mosaic - 0.5 # 0 -1 -> -0.5 to 0.5
        rgb_mosaic = rgb_mosaic.astype(np.float32)[np.newaxis, ...]
        rgb_demosaiced = rgb_demosaiced - 0.5   # 0 -1 -> -0.5 to 0.5
        rgb_demosaiced = np.transpose(rgb_demosaiced, (2,0,1)).astype(np.float32)
        

        return ms_mosaic, ms_demosaiced, rgb_mosaic, rgb_demosaiced, ms_mean, rgb_mean

    def __getitem__(self, idx):
        # Load multi-spectral and RGB files
        ms_file = self.ms_files[idx]
        rgb_file = self.rgb_files[idx]  
        #replace iso400 with iso100
        gt_ms_file = ms_file.replace("iso400", "iso100")
        gt_rgb_file = rgb_file.replace("iso400", "iso100")
        

        ms_img = np.load(ms_file)  # 256*256*16
        rgb_img = np.load(rgb_file)  # 256*256*3
        gt_ms_img = np.load(gt_ms_file)  # 256*256*16
        gt_rgb_img = np.load(gt_rgb_file)  # 256*256*3

        #Convert 16-bit to 0-1 range
        ms_img = ms_img / 65535.0
        rgb_img = rgb_img / 65535.0
        gt_ms_img = gt_ms_img / 65535.0
        gt_rgb_img = gt_rgb_img / 65535.0

        #find the view number
        view_number = re.search(r'view(\d+)', ms_file).group(1)
        
        
        _, gt_ms_demosaiced,_,gt_rgb_demosaiced,_,_ = self.preprocess(gt_ms_img, gt_rgb_img)
        
        output_img = gt_ms_demosaiced
        
        
        if self.input_type == 'ms_mosaic':
            ms_mosaic, ms_demosaiced_noise, rgb_mosaic, rgb_demosaiced,_,_ = self.preprocess(ms_img, rgb_img)
            input_img = ms_mosaic
            output_img = gt_ms_demosaiced
        elif self.input_type == 'ms_mosaic_ds4':
            ms_mosaic, _, rgb_mosaic, rgb_demosaiced,_,_ = self.preprocess(ms_img, rgb_img, ds_factor=4)
            input_img = ms_mosaic
            output_img = gt_ms_demosaiced
        elif self.input_type == 'rgb_mosaic_view_2':
            ms_mosaic, _, rgb_mosaic, _,_,_ = self.preprocess(ms_img, rgb_img)
            
            input_img = rgb_mosaic
            output_img = gt_rgb_demosaiced
        
        elif self.input_type in ['ms_mosaic_inline', 'rgb_mosaic_inline_w_ms', 'rgb_mosaic_w_fullms_inline', 
                                 'ms_mosaic_inline_w_rgb', 'ms_mosaic_ds4_inline_w_rgb']:
        
            
            if self.input_type == 'ms_mosaic_inline':
                ms_mosaic, _, rgb_mosaic, rgb_demosaiced,_,_ = self.preprocess(ms_img, rgb_img)
                input_img = np.concatenate((ms_mosaic, rgb_mosaic), axis=0)
            elif self.input_type == 'ms_mosaic_inline_w_rgb':
                ms_mosaic, _, rgb_mosaic_other_view, rgb_demosaiced_other_view,_,_ = self.preprocess(ms_img, rgb_img)
                input_img = np.concatenate((ms_mosaic, rgb_mosaic_other_view), axis=0)
                output_img = gt_ms_demosaiced
            elif self.input_type == 'ms_mosaic_ds4_inline_w_rgb':
                ms_mosaic, _, rgb_mosaic_other_view, rgb_demosaiced_other_view,_,_ = self.preprocess(ms_img, rgb_img, ds_factor=4)
                #find size of rgb_img
                C, H, W = rgb_demosaiced_other_view.shape
                #make empty array of size H,W
                ms_mosaic_embed = np.zeros((1,H,W)) 
                ms_mosaic_embed[:, :H//4, :W//4] = ms_mosaic #top left
                
                input_img = np.concatenate((ms_mosaic_embed, rgb_mosaic_other_view), axis=0)
                output_img = gt_ms_demosaiced
        else:
            raise ValueError("Unknown input type")

        
        out = {'in': input_img, 'out': output_img}
            
        #add scene number to the output
        scene_number = re.search(r'scene(\d+)', ms_file).group(1)
        position_number = re.search(r'position(\d+)', ms_file).group(1)
        out['scene_number'] = scene_number
        out['view_number'] = view_number
        out['position_number'] = position_number
        return out



