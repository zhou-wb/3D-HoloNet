import prop_ideal
import torch
import torch.nn as nn
from load_flying3d import FlyingThings3D_loader
# from load_hypersim import hypersim_TargetLoader
from image_loader import TargetLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import utils
from algorithm import double_phase, gradient_descent
from propagation_ASM import propagation_ASM
# from forward_backward_propagation import threeD_dpac
from load_CNNpropCNN.load_CNNpropCNN import load_CNNpropCNN
import numpy as np
import imageio.v3 as iio
import os
from PIL import Image
from torchinfo import summary


#####################
# Optics Parameters #
#####################

# distance between the reference(middle) plane and slm
prop_dist = 0.10
# prop_dist = 0.156
backward_prop_dist = -0.1049
# distance between the reference(middle) plane and all the 8 target planes
# prop_dists_from_wrp = [-0.0021, -0.0014, -0.0007, -0.0000, 0.0007, 0.0014, 0.0021, 0.0028]
prop_dists_from_wrp = [0.0000, 0.0007, 0.0014, 0.0021, 0.0028, 0.0035, 0.0042, 0.0049]
# depth in diopter space (m^-1) to compute the masks for rgbd input
virtual_depth_planes = [0.0, 0.08417508417508479, 0.14124293785310726, 0.24299599771297942, 0.3171856978085348, 0.4155730533683304, 0.5319148936170226, 0.6112104949314254]
plane_idx = [0, 1, 2, 3, 4, 5, 6, 7]
prop_dists_from_wrp = [prop_dists_from_wrp[idx] for idx in plane_idx]
virtual_depth_planes = [virtual_depth_planes[idx] for idx in plane_idx]
channel = 1
wavelength_rgb = [6.379e-07, 5.249e-07, 4.435e-07]
wavelength = wavelength_rgb[channel]
feature_size = (8.0e-06, 8.0e-06)
F_aperture = 1
device = torch.device('cuda:0')
loss_fn = nn.MSELoss().to(device)
image_res = (1080, 1920)
roi_res = (880, 1600)

# propagation
forward_ASM = []
for prop_dist_ in prop_dists_from_wrp:
    forward_ASM.append(prop_ideal.SerialProp(prop_dist_, wavelength, feature_size,
                            'ASM', F_aperture, dim=1).to(device))
forward_prop_without_aperture = prop_ideal.SerialProp(prop_dist, wavelength, feature_size,
                                        'ASM',  F_aperture, prop_dists_from_wrp=prop_dists_from_wrp,
                                        dim=1).to(device)
forward_prop_with_aperture = prop_ideal.SerialProp(prop_dist, wavelength, feature_size,
                                        'ASM',  F_aperture=0.5, prop_dists_from_wrp=prop_dists_from_wrp,
                                        dim=1).to(device)
backward_ASM = prop_ideal.SerialProp(backward_prop_dist, wavelength, feature_size,
                            'ASM', F_aperture, dim=1).to(device)
cnnpropcnn = load_CNNpropCNN().to(device)
# summary(cnnpropcnn, (8, 1080, 1920))
holo_path = {}
holo_path['frame_1_with_tv'] = '3D-HoloNet_model.pth'

def cond_loader(data_name, channel, image_res, roi_res, virtual_depth_planes):
    random_seed = None #random_seed = None for not shuffle
    return_type = 'image_mask_depth_id'
    if data_name == 'flying3d':
        data_path = '/mnt/ssd1/feifan/data/flying3d/'
        loader = FlyingThings3D_loader(data_path=data_path,
                                       channel=channel, image_res=image_res, roi_res=roi_res,
                                     virtual_depth_planes=virtual_depth_planes,
                                     return_type=return_type,
                                     random_seed=random_seed,
                                     slice=(0.9,1),
                                     keyword='left/0006',
                                     )
    elif data_name == 'div2k':
        data_path = 'div2k'
        loader = FlyingThings3D_loader(data_path=data_path, channel=channel,
                           image_res=image_res, roi_res=roi_res, 
                           virtual_depth_planes=virtual_depth_planes,
                           return_type=return_type,
                            random_seed=random_seed,
                           )
    elif data_name == 'grass':
        data_path = 'data'
        loader = TargetLoader(data_path=data_path, target_type='rgbd', channel=channel,
                           image_res=image_res, roi_res=roi_res, 
                           virtual_depth_planes=virtual_depth_planes,
                           )
    elif data_name == 'usaf':
        data_path = 'usaf'
        loader = TargetLoader(data_path=data_path, target_type='rgbd', channel=channel,
                           image_res=image_res, roi_res=roi_res, 
                           virtual_depth_planes=virtual_depth_planes,
                           )
    elif data_name == 'usaf_sparse':
        data_path = 'usaf_sparse'
        loader = FlyingThings3D_loader(data_path=data_path, channel=channel,
                           image_res=image_res, roi_res=roi_res, 
                           virtual_depth_planes=virtual_depth_planes,
                           return_type=return_type,
                            random_seed=random_seed,
                           )
    loader = DataLoader(loader, batch_size=1)

    return loader


def threeD_dpac(imgs, masks, roi_res, forward_ASM, backward_ASM, device):
    masked_imgs = imgs * masks
    initial_phase = torch.zeros_like(masked_imgs).to(device)  
    initial_complex_field = torch.polar(masked_imgs, initial_phase)
    for i in range(1, initial_complex_field.shape[1]):
        initial_phase[:,i] = (forward_ASM[i](initial_complex_field[:,i]) * masks[:,i]).angle()
    complex_field = torch.polar(masked_imgs, initial_phase)
    for i in range(initial_complex_field.shape[1] - 1):
        complex_field[:,i+1] = forward_ASM[1](complex_field[:,i]) * (1 - masks[:,i+1]) + complex_field[:,i+1] 
    slm_complex_field = backward_ASM(complex_field[:,-1])
    final_phase = double_phase(slm_complex_field, three_pi=False, mean_adjust=True)
    return final_phase


def sgd(target_amp, target_mask, device, propagator, roi_res=(880, 1600), num_iters=1000):
    target_amp = target_amp.to(device).detach()
    if target_mask is not None:
        target_mask = target_mask.to(device).detach()
    if len(target_amp.shape) < 4:
        target_amp = target_amp.unsqueeze(0)
    init_phase_range = 1.0
    init_phase = (init_phase_range * (-0.5 + 1.0 * torch.rand(1, 1, * (1080, 1920)))).to(device)
    
    # 可改参数 loss_func lr
    results = gradient_descent(init_phase, target_amp, target_mask,
                            forward_prop=propagator, num_iters=num_iters, roi_res=roi_res ,
                            loss_fn=nn.functional.l1_loss, lr=0.02,)
    
    final_phase = results['final_phase']
    return final_phase


def holonet(imgs, masks, device, model_path):
    inverse_prop = torch.load(model_path).to(device)
    masked_imgs = imgs * masks
    final_phase = inverse_prop(masked_imgs)
    # summary(inverse_prop, (1, 8, 1080, 1920))
    # print(inverse_prop)
    
    return final_phase

def cond_mkdir(output_path, sub_path):
    path = os.path.join(os.path.join(output_path, sub_path))
    if not os.path.exists(path):
        os.makedirs(path)
    return path



num_iters = 1000
# for data_name in ['flying3d', 'div2k', 'grass']:
for data_name in [ 'grass']:
# for data_name in ['usaf_sparse']:
    loader = cond_loader(data_name, channel, image_res, roi_res, virtual_depth_planes)
    final_phase, recon_amp, all_in_focus, mapped_final_phase, paths = {}, {}, {}, {}, {}
    output_folder = './experiments_holonet/' + data_name + '/green'
    for path in ['img', 'depth', 'depth_color', 'mask', 'masked_img', 'recon']:
        paths[path] = cond_mkdir(output_folder, path)
        
    count = 0
    for imgs_masks_depth_id in tqdm(loader): 
        count += 1
        if count == 2 and data_name == 'flying3d':
            break
        
        imgs, masks, depth, imgs_id = imgs_masks_depth_id
        
        if data_name == 'usaf':
            masks = 1 - masks
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        print(imgs_id[0])
        sub_name = imgs_id[0].replace('/', '_')
        crop_masks = utils.crop_image(masks, roi_res, stacked_complex=False)

        with torch.no_grad():
            for key in holo_path:
                final_phase[key] = holonet(imgs, masks, device, model_path=holo_path[key])
            for key in final_phase:
                if isinstance(final_phase[key], tuple):
                    phase_0, phase_1 = final_phase[key]
                    recon_amp[key] = (cnnpropcnn(phase_0).abs() + cnnpropcnn(phase_1).abs()) / 2
                    mapped_final_phase_0 = (1. - ((phase_0 + np.pi) % (2 * np.pi)) / (2 * np.pi)) * 255
                    mapped_final_phase_1 = (1. - ((phase_1 + np.pi) % (2 * np.pi)) / (2 * np.pi)) * 255
                    iio.imwrite(os.path.join(output_folder, f'{sub_name}_{key}_slm_phase_0.png'), mapped_final_phase_0.squeeze(0).detach().cpu().numpy().round().astype(np.uint8))
                    iio.imwrite(os.path.join(output_folder, f'{sub_name}_{key}_slm_phase_1.png'), mapped_final_phase_1.squeeze(0).detach().cpu().numpy().round().astype(np.uint8))
                else:
                    recon_amp[key] = cnnpropcnn(final_phase[key]).abs()
                    mapped_final_phase[key] = (1. - ((final_phase[key] + np.pi) % (2 * np.pi)) / (2 * np.pi)) * 255
                    re_map_phase = (2*np.pi * (1 - mapped_final_phase[key] / 255.0)) - np.pi
                    iio.imwrite(os.path.join(output_folder, f'{sub_name}_{key}_slm_phase.png'), mapped_final_phase[key].squeeze(0).detach().cpu().numpy().round().astype(np.uint8))
            
                recon_amp[key] = (recon_amp[key] - recon_amp[key].min()) / (recon_amp[key].max() - recon_amp[key].min())

                all_in_focus[key] = torch.sum(recon_amp[key] * crop_masks, dim=1)
            

            cv2.imwrite(os.path.join(paths['img'], f'{sub_name}_image.png'), (imgs.squeeze(0).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8))
            

            masked_imgs = imgs * masks
            for i in range(len(virtual_depth_planes)):
                cv2.imwrite(os.path.join(paths['masked_img'], f'{sub_name}_masked_img_{i}.png'), (masked_imgs.squeeze(0)[i].cpu().numpy()*255).astype(np.uint8))
                    
                for i in range(len(virtual_depth_planes)):
                    cv2.imwrite(os.path.join(paths['recon'], f'{sub_name}_{key}_recon_{i}.png'), (recon_amp[key].squeeze(0)[i].detach().cpu().numpy()*255).astype(np.uint8))
        
        
        
def phasemap_8bit(phasemap, inverted=True):
    """Convert a phasemap tensor into a numpy 8bit phasemap for SLM
    Input
    -----
    :param phasemap: input phasemap tensor, in the range of [-pi, pi].
    :param inverted: flag for if the phasemap is inverted.
    Output
    ------
    :return phase_out_8bit: output phasemap, with uint8 dtype (in [0, 256))
    """

    out_phase = phasemap.cpu().detach().squeeze().numpy()
    out_phase = ((out_phase + np.pi) % (2 * np.pi)) / (2 * np.pi)
    if inverted:
        phase_out_8bit = ((1 - out_phase) * 256).round().astype(np.uint8) # change from 255 to 256
    else:
        phase_out_8bit = ((out_phase) * 256).round().astype(np.uint8)

    return phase_out_8bit