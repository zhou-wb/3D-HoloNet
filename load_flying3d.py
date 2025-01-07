import os, copy
from imageio import imread
from skimage.transform import resize
from torchvision.transforms.functional import resize as resize_tensor
import cv2
import random
import numpy as np
import torch
from zmq import device
import utils
import matplotlib.pyplot as plt
from PIL import Image


cv2.setNumThreads(0)

# type is a string that appeared in the image file name, should be chosen from 'color' or 'depth'
def get_last_dir_paths(root_path):
    last_dir_paths = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        # 如果该文件夹没有子文件夹，则保存该文件夹路径
        if not dirnames:
            last_dir_paths.append(os.path.relpath(dirpath, './'))
    
    return last_dir_paths


def get_image_filenames(dir, keyword=None):
    """Returns all files in the input directory dir that are images"""

    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    if keyword is not None:
        file_list = [f for f in file_list if keyword in f]
    return file_list



def resize_keep_aspect(image, target_res, pad=False, lf=False, pytorch=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False, lf=lf)

    # switch to numpy channel dim convention, resize, switch back
    if lf or pytorch:
        image = resize_tensor(image, target_res)
        return image
    else:
        image = np.transpose(image, axes=(1, 2, 0))
        image = resize(image, target_res, mode='reflect')
        return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res, pytorch=False):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=pytorch, stacked_complex=False),
                            target_res, pytorch=pytorch, stacked_complex=False)




class FlyingThings3D_loader(torch.utils.data.IterableDataset):
    """Loads target amp/mask tuples for phase optimization

    Class initialization parameters
    -------------------------------
    :param data_path:
    :param channel:
    :param image_res:
    :param roi_res:
    :param crop_to_roi:
    :param shuffle:
    :param virtual_depth_planes:
    :param return_type: 'image_mask_id' or 'image_depth_id'

    """

    def __init__(self, data_path, channel=None,
                 image_res=(1080, 1920), roi_res=(960, 1680),
                 crop_to_roi=False, scale_vd_range=True,
                 virtual_depth_planes=None, return_type='image_mask_id',
                 random_seed=None, slice=(0, 1), keyword=None):
        """ initialization """
        if isinstance(data_path, str) and not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')

        assert(slice[0] >=0 and slice[0]<=1 and slice[1] >=0 and slice[1]<=1 and slice[0] < slice[1]) 
        
        self.data_path = data_path
        # self.target_type = target_type.lower()
        self.channel = channel
        self.roi_res = roi_res
        # self.crop_to_roi = crop_to_roi
        self.image_res = image_res
        # self.physical_depth_planes = physical_depth_planes
        self.virtual_depth_planes = virtual_depth_planes
        self.scale_vd_range = scale_vd_range
        self.vd_min = 0.01
        self.vd_max = max(self.virtual_depth_planes)
        self.return_type = return_type
        self.slice = slice
        self.random_seed = random_seed
        self.keyword = keyword
        
        # self.im_names = get_image_filenames(dir = os.path.join(self.data_path, 'scene_cam_00_final_hdf5'), keyword = 'color')
        self.im_names = get_image_filenames(dir = os.path.join(self.data_path, 'frames_cleanpass'), keyword=self.keyword)
        self.depth_names = get_image_filenames(dir = os.path.join(self.data_path, 'disparity'), keyword=self.keyword)
        
        length = len(self.im_names)
        assert(len(self.im_names) == len(self.depth_names))

        self.im_names.sort()
        self.depth_names.sort()

        self.order = list((i) for i in range(length))
        if self.random_seed != None:
            random.Random(self.random_seed).shuffle(self.order)
        self.order = self.order[round(self.slice[0]*length):round(self.slice[1]*length)]

    def __iter__(self):
        self.ind = 0            
        return self

    def __len__(self):
        return len(self.order)

    def __next__(self):
        if self.ind < len(self.order):
            img_idx = self.order[self.ind]
            self.ind += 1
            if self.return_type == 'image_mask_id':
                return self.load_image_mask(img_idx)
            elif self.return_type == 'image_depth_id':
                return self.load_image_depth(img_idx)
            elif self.return_type == 'image_mask_focalstack_id':
                return self.load_image_mask_focalstack(img_idx)
            elif self.return_type == 'image_mask_depth_id':
                return self.load_image_mask_depth(img_idx)
            elif self.return_type == 'image_mask':
                return self.load_image_mask_without_id(img_idx)
            
        else:
            raise StopIteration
        
    def rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def load_image(self, filenum):
        # with h5py.File(self.im_names[filenum], 'r') as im_file:
        #     # im_file = h5py.File(self.im_names[filenum])
        #     im = im_file['dataset'][:]
        
        # im = im[..., self.channel, np.newaxis]
        # im = utils.im2float(im, dtype=np.float64)
        # im = np.transpose(im, axes=(2, 0, 1))
        
        im = imread(self.im_names[filenum])
        
        # if self.channel:    
        #     im = im[..., self.channel, np.newaxis]
        im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1
        im = self.rgb2gray(im)[..., np.newaxis]
        # linearize intensity and convert to amplitude
        im = utils.srgb_gamma2lin(im)
        im = np.sqrt(im)  # to amplitude

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))
        
        im = resize_keep_aspect(im, self.roi_res)
        im = pad_crop_to_res(im, self.image_res)
        
        path = os.path.splitext(self.im_names[filenum])[0]

        return (torch.from_numpy(im).float(),
                None,
                os.path.join(*path.split(os.sep)[-6:])) # get the path after flying3d folder. e.g. frames_cleanpass/TRAIN/C/0212/left/0007
        
    def depth_convert(self, depth):
        # NaN to inf
        depth[depth==0] = 1.0
        # convert to double
        # depth = depth.double()
        
        # meter to diopter conversion
        # depth = 1 / (depth + 1e-20)        
        return depth

    def load_depth(self, filenum):
        depth_path = self.depth_names[filenum]
        if depth_path.endswith('.pfm'):
            depth = utils.read_pfm(depth_path)
        elif depth_path.endswith('.png'):
            depth = imread(depth_path)
    
        depth = depth.astype(np.float64)  # convert to double, max 1

        # distance = self.depth_convert(distance)

        # convert from numpy array to pytorch tensor, shape = (1, original_h, original_w)
        depth = torch.from_numpy(depth.copy()).float().unsqueeze(0)
        
        depth = resize_keep_aspect(depth, self.roi_res, pytorch=True)
        
        
        # perform scaling in meters
        if self.scale_vd_range:
            depth = depth - depth.min()
            depth = (depth / depth.max()) * (self.vd_max - self.vd_min)
            depth = depth + self.vd_min
        
        # depth_value = depth.flatten()
        # depth_value_sort, index_sort = depth_value.sort(stable=True)
        # depth_value_

        depth = pad_crop_to_res(depth, self.image_res, pytorch=True)

        # check nans
        if (depth.isnan().any()):
            print("Found Nans in target depth!")
            min_substitute = self.vd_min * torch.ones_like(depth)
            depth = torch.where(depth.isnan(), min_substitute, depth)

        path = os.path.splitext(self.depth_names[filenum])[0]

        return (depth.float(),
                None,
                os.path.join(*path.split(os.sep)[-6:])) # get the path after flying3d folder. e.g. disparity/TRAIN/C/0212/left/0007
    
    def load_focalstack(self, filenum):
        focalstack_root = '/mnt/ssd1/feifan/data/flying3d_focalstack'
        img_path = os.path.join(*os.path.splitext(self.im_names[filenum])[0].split(os.sep)[-6:])
        focalstack_path = os.path.join(focalstack_root, img_path, f"focalstack_{['r','g','b'][self.channel]}.pth")
        focalstack = torch.load(focalstack_path).squeeze()
        
        focalstack = resize_keep_aspect(focalstack, self.roi_res, pytorch=True)
        focalstack = pad_crop_to_res(focalstack, self.image_res, pytorch=True)

        return focalstack

    def load_image_mask(self, filenum):
        img_none_idx = self.load_image(filenum)
        depth_none_idx = self.load_depth(filenum)
        mask = utils.decompose_depthmap_equal(pad_crop_to_res(depth_none_idx[0], self.roi_res), 
                                              self.virtual_depth_planes)
        mask = pad_crop_to_res(mask, self.image_res)
        return (img_none_idx[0], mask, img_none_idx[-1])
    
    def load_image_mask_without_id(self, filenum):
        img_none_idx = self.load_image(filenum)
        depth_none_idx = self.load_depth(filenum)
        mask = utils.decompose_depthmap_equal(pad_crop_to_res(depth_none_idx[0], self.roi_res), 
                                              self.virtual_depth_planes)
        mask = pad_crop_to_res(mask, self.image_res)
        return (img_none_idx[0], mask)
    
    def load_image_depth(self, filenum):
        img_none_idx = self.load_image(filenum)
        depth_none_idx = self.load_depth(filenum)
        return (img_none_idx[0], depth_none_idx[0], img_none_idx[-1])
    
    def load_image_mask_focalstack(self, filenum):
        img_none_idx = self.load_image(filenum)
        depth_none_idx = self.load_depth(filenum)
        mask = utils.decompose_depthmap(depth_none_idx[0], self.virtual_depth_planes)
        focalstack = self.load_focalstack(filenum)
        return (img_none_idx[0], mask, focalstack, img_none_idx[-1])

    def load_image_mask_depth(self, filenum):
        img_none_idx = self.load_image(filenum)
        depth_none_idx = self.load_depth(filenum)
        mask = utils.decompose_depthmap(depth_none_idx[0], self.virtual_depth_planes)
        return (img_none_idx[0], mask, depth_none_idx[0], img_none_idx[-1])
    