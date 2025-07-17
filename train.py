import sys
sys.path.append('.')
from custom_loss import calculate_image_entropy
from fft import real_fourier_transform

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure




import numpy as np
import torch, os, random, utils, time
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# propagation network related 
from inverse3d_prop import InversePropagation
from prop_model import CNNpropCNN_default
import prop_ideal

# dataset related
from load_flying3d import FlyingThings3D_loader

from load_CNNpropCNN.load_CNNpropCNN import load_CNNpropCNN

from custom_loss import total_variation_loss

#####################
# Optics Parameters #
#####################

# distance between the reference(middle) plane and slm
prop_dist = 0.1021
# distance between the reference(middle) plane and all the 8 target planes 
prop_dists_from_wrp = [-0.0021, -0.0014, -0.0007, -0.0000, 0.0007, 0.0014, 0.0021, 0.0028]
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

#######################
# Training Parameters #
#######################
device = torch.device('cuda:1')
loss_fn = nn.MSELoss().to(device)
cal_psnr = PeakSignalNoiseRatio().to(device)
cal_ssim = StructuralSimilarityIndexMeasure().to(device)

learning_rate = 1e-4
max_epoch = 100000

loss_type_list = ['in-focus-loss', 'all-image-loss']
loss_id = 0
loss_type_config = loss_type_list[loss_id]

if loss_type_config == 'in-focus-loss':
    return_type = 'image_mask_id'
elif loss_type_config == 'all-image-loss':
    return_type = 'image_mask_focalstack_id'
# If there are nan in output, consider enable this to debug
# torch.autograd.set_detect_anomaly(True)

######################
# Dataset Parameters #
######################

batch_size = 1
dataset_list = ['Hypersim', 'FlyingThings3D', 'MitCGH']
dataset_id = 1
dataset_name = dataset_list[dataset_id]
# resize_to_1080p = True
# for_uformer = False

image_res = (1080, 1920)
roi_res = (880, 1600)
random_seed = 10 #random_seed = None for not shuffle

if dataset_name == 'FlyingThings3D':
    data_path = 'xxx/data/flying3d/'
    train_loader = FlyingThings3D_loader(data_path=data_path,
                                         channel=channel, image_res=image_res, roi_res=roi_res,
                                         virtual_depth_planes=virtual_depth_planes,
                                         return_type=return_type,
                                         random_seed=random_seed,
                                         slice=(0,0.8),
                                         keyword = 'left/0006',
                                         )
    val_loader = FlyingThings3D_loader(data_path=data_path,
                                       channel=channel, image_res=image_res, roi_res=roi_res,
                                       virtual_depth_planes=virtual_depth_planes,
                                       return_type=return_type,
                                       random_seed=random_seed,
                                       slice=(0.8,0.9),
                                       keyword = 'left/0006',
                                       )
else:
    raise ValueError(f"Dataset: '{dataset_name}' Not Implement!")
print(f"train set length: {len(train_loader)}")
print(f"val  set length: {len(val_loader)}")
train_dataloader = DataLoader(train_loader, batch_size=batch_size)
val_dataloader = DataLoader(val_loader, batch_size=batch_size)

####################################
# Load Networks -- Inverse Network #
####################################

# choose the network structure by set the config_id to 0,1,2
inverse_network_list = ['unetasmunet',]
network_id = 0
inverse_network_config = inverse_network_list[network_id]
inverse_prop = InversePropagation(inverse_network_config, prop_dists_from_wrp=prop_dists_from_wrp, prop_dist=prop_dist,
                                  wavelength=wavelength, feature_size=feature_size, device=device, F_aperture=F_aperture,
                                  image_res=image_res)
inverse_prop = inverse_prop.to(device)
optimizer = torch.optim.Adam(inverse_prop.parameters(), lr=learning_rate)

####################################
# Load Networks -- Forward Network #
####################################

forward_prop_list = ['ASM', 'CNNpropCNN']
forward_prop_id = 1
forward_prop_config = forward_prop_list[forward_prop_id]

if forward_prop_config == 'CNNpropCNN':
    forward_network_config = 'CNNpropCNN'
    forward_prop = load_CNNpropCNN()
    forward_prop = forward_prop.to(device)
    for param in forward_prop.parameters():
        param.requires_grad = False
elif forward_prop_config == 'ASM':
    forward_network_config = 'ASM'
    forward_prop = prop_ideal.SerialProp(prop_dist, wavelength, feature_size,
                                        'ASM', F_aperture, prop_dists_from_wrp,
                                        dim=1)
    forward_prop = forward_prop.to(device)
else:
    raise ValueError(f"Forward network: '{forward_prop_config}' Not Implement!")

################
# Init metrics #
################

total_train_step = 0
best_val_loss = float('inf')
best_test_psnr = 0

# init tensorboard
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
run_id = dataset_name + ['r','g','b'][channel] + '-' + inverse_network_config + '-' + \
    str(learning_rate) + '-' + forward_network_config + '-' + \
    f'{image_res[0]}_{image_res[1]}-{roi_res[0]}_{roi_res[1]}' + '-' + \
    f'{len(plane_idx)}_target_planes' + '-' + loss_type_config #+ str(defocus_weight)
print('Dataset:', dataset_name)
print('Channel:', ['red','green','blue'][channel])
print('Inverse Network:', inverse_network_config)
print('Forward Prop:', forward_network_config)
print('Learning Rate:', learning_rate)
print('Image Resolution:', image_res)
print('ROI Resolution:', roi_res)
print('Batch Size:', batch_size)
print('Number of Target Planes:', len(plane_idx))
print('Loss Type:', loss_type_config)
print('The Network will be Trained on:', torch.cuda.get_device_name(device))
input("Press Enter to continue...")
run_folder_name = time_str + '-' + run_id
writer = SummaryWriter(f'runs/{run_folder_name}')
writer.add_scalar("ratio/learning_rate", learning_rate)

#################
# Training loop #
#################

start_events = torch.cuda.Event(enable_timing=True)
inverse_events = torch.cuda.Event(enable_timing=True)

for i in range(max_epoch):
    print(f"----------Training Start (Epoch: {i+1})-------------")
    total_train_loss, total_train_psnr = 0, 0
    train_items_count = 0
    # training steps
    inverse_prop.train()
    average_scale_factor = 0
    for imgs_masks_id in train_dataloader:

        if total_train_step == 10:
            sdf = 1
        
        if loss_type_config == 'in-focus-loss':
            imgs, masks, imgs_id = imgs_masks_id
        elif loss_type_config == 'all-image-loss':
            imgs, masks, focalstack, imgs_id = imgs_masks_id
            focalstack = focalstack.to(device)

        if imgs.max() <= 0.1:
            continue

        imgs = imgs.to(device)
        masks = masks.to(device)
        masked_imgs = imgs * masks

        # inverse propagation
        slm_phase_0 = inverse_prop(masked_imgs)
        # forward propagation
        outputs_field_0 = forward_prop(slm_phase_0)
        # outputs_field_1 = forward_prop(slm_phase_1)
        outputs_amp_0 = outputs_field_0.abs()
        outputs_amp = outputs_amp_0
        # outputs_amp_1 = outputs_field_1.abs()
        # outputs_amp = (outputs_amp_0 + outputs_amp_1) / 2
        
        imgs = utils.crop_image(imgs, roi_res, stacked_complex=False)
        masks = utils.crop_image(masks, roi_res, stacked_complex=False) # need to check if process before network
        outputs_amp = utils.crop_image(outputs_amp, roi_res, stacked_complex=False)
        crop_slm_phase_0 = utils.crop_image(slm_phase_0, roi_res, stacked_complex=False)
        # crop_slm_phase_1 = utils.crop_image(slm_phase_1, roi_res, stacked_complex=False)
        
        if loss_type_config == 'in-focus-loss':
            final_amp = torch.sum(outputs_amp * masks, dim=1)

            with torch.no_grad():
                s = (final_amp * imgs).mean() / \
                    (final_amp ** 2).mean()  # scale minimizing MSE btw recon and target
                average_scale_factor += s
                
            mse_loss = loss_fn(s * final_amp, imgs)
            entropy_loss_0 = calculate_image_entropy(crop_slm_phase_0)
            # entropy_loss_1 = calculate_image_entropy(crop_slm_phase_1)
            tv_loss_0 = total_variation_loss(crop_slm_phase_0, 1e-4)
            # tv_loss_1 = total_variation_loss(crop_slm_phase_1, 1e-4)
            loss = mse_loss + tv_loss_0
        
        writer.add_scalar("ratio/scale", s, total_train_step)
        
        with torch.no_grad(): 
            psnr = cal_psnr(s * final_amp.unsqueeze(0), imgs)
            ssim = cal_ssim(s * final_amp.unsqueeze(0), imgs)
        
        total_train_loss += loss.item()
        total_train_psnr += psnr.item()
        writer.add_scalar("loss/train_loss", loss.item(), total_train_step)
        writer.add_scalar("loss/mse_loss", mse_loss.item(), total_train_step)
        writer.add_scalar("loss/tv_loss_0", tv_loss_0, total_train_step)
        # writer.add_scalar("loss/tv_loss_1", tv_loss_1, total_train_step)
        writer.add_scalar("loss/entropy_loss_0", entropy_loss_0.item(), total_train_step)
        # writer.add_scalar("loss/entropy_loss_1", entropy_loss_1.item(), total_train_step)
        writer.add_scalar("metrics/train_psnr", psnr.item(), total_train_step)
        writer.add_scalar("metrics/train_ssim", ssim.item(), total_train_step)
        
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        train_items_count += 1
        if (total_train_step) % 100 == 0:
            
            print(f"Training Step {total_train_step}, Loss: {loss.item()}")
            
            if (total_train_step) % 1000 == 0:
                mapped_slm_phase_0 = 1. - ((slm_phase_0 + np.pi) % (2 * np.pi)) / (2 * np.pi)
                # mapped_slm_phase_1 = 1. - ((slm_phase_1 + np.pi) % (2 * np.pi)) / (2 * np.pi)
                crop_mapped_slm_phase_0 = utils.crop_image(mapped_slm_phase_0, roi_res, stacked_complex=False)
                # crop_mapped_slm_phase_1 = utils.crop_image(mapped_slm_phase_1, roi_res, stacked_complex=False)

                writer.add_image(f'train_phs/phase_0', mapped_slm_phase_0[0,0], total_train_step, dataformats='HW')
                # writer.add_image(f'train_phs/phase_1', mapped_slm_phase_1[0,0], total_train_step, dataformats='HW')
                writer.add_histogram(f'train_phs/phase_hist_0', mapped_slm_phase_0[0,0], total_train_step)
                # writer.add_histogram(f'train_phs/phase_hist_1', mapped_slm_phase_1[0,0], total_train_step)
                writer.add_image(f"train_phs/phase_fft_0", real_fourier_transform(crop_mapped_slm_phase_0[0,0]), total_train_step, dataformats='HW')
                # writer.add_image(f"train_phs/phase_fft_1", real_fourier_transform(crop_mapped_slm_phase_1[0,0]), total_train_step, dataformats='HW')
                
                for j in range(len(plane_idx)):
                    writer.add_image(f'train_input_images/plane{j}', masked_imgs[0,j], total_train_step, dataformats='HW')
                    writer.add_images(f'train_output_images/plane{j}', s*outputs_amp[0,j], total_train_step, dataformats='HW')

                writer.flush()
    
    average_train_loss = total_train_loss/train_items_count
    average_train_psnr = total_train_psnr/train_items_count
    average_scale_factor = average_scale_factor/train_items_count
    writer.add_scalar("loss/average_train_loss", average_train_loss, total_train_step)
    writer.add_scalar("metrics/average_train_psnr", average_train_psnr, total_train_step)
    writer.add_scalar("ratio/average_scale_factor", average_scale_factor, total_train_step)
    
    ###################
    # Validation loop #
    ###################
    # test the model on validation set after every epoch
    inverse_prop.eval()
    total_val_loss = 0
    total_val_psnr = 0
    total_val_ssim = 0
    val_items_count = 0
    total_inference_time = 0
    record_id = random.randrange(len(val_loader)//batch_size)
    # record_id = 0
    with torch.no_grad():
        for imgs_masks_id in val_dataloader: 
            if loss_type_config == 'in-focus-loss':
                imgs, masks, imgs_id = imgs_masks_id

            imgs = imgs.to(device)
            masks = masks.to(device)
            masked_imgs = imgs * masks
            
            # inverse propagation
            start_events.record()
            slm_phase_0 = inverse_prop(masked_imgs)
            
            inverse_events.record()
            torch.cuda.synchronize()
            inference_time = start_events.elapsed_time(inverse_events)
            total_inference_time += inference_time
            # forward propagation
            outputs_field_0 = forward_prop(slm_phase_0)
            # outputs_field_1 = forward_prop(slm_phase_1)
            outputs_amp_0 = outputs_field_0.abs()
            outputs_amp = outputs_amp_0
            # outputs_amp_1 = outputs_field_1.abs()
            # outputs_amp = (outputs_amp_0 + outputs_amp_1) / 2
            
            imgs = utils.crop_image(imgs, roi_res, stacked_complex=False)
            masks = utils.crop_image(masks, roi_res, stacked_complex=False) # need to check if process before network
            outputs_amp = utils.crop_image(outputs_amp, roi_res, stacked_complex=False)
            masked_imgs = utils.crop_image(masked_imgs, roi_res, stacked_complex=False)
            crop_slm_phase_0 = utils.crop_image(slm_phase_0, roi_res, stacked_complex=False)
            # crop_slm_phase_1 = utils.crop_image(slm_phase_1, roi_res, stacked_complex=False)
            
            if loss_type_config == 'in-focus-loss':
                final_amp = torch.sum(outputs_amp * masks, dim=1)
                
                mse_loss = loss_fn(average_scale_factor * final_amp, imgs)
                entropy_loss_0 = calculate_image_entropy(crop_slm_phase_0)
                # entropy_loss_1 = calculate_image_entropy(crop_slm_phase_1)
                tv_loss_0 = total_variation_loss(crop_slm_phase_0, 1e-4)
                # tv_loss_1 = total_variation_loss(crop_slm_phase_1, 1e-4)
                loss = mse_loss 
            
            with torch.no_grad(): 
                psnr = cal_psnr(average_scale_factor * final_amp.unsqueeze(0), imgs)
                ssim = cal_ssim(average_scale_factor * final_amp.unsqueeze(0), imgs)
            
            if val_items_count == record_id:
                mapped_slm_phase_0 = 1. - ((slm_phase_0 + np.pi) % (2 * np.pi)) / (2 * np.pi)
                # mapped_slm_phase_1 = 1. - ((slm_phase_1 + np.pi) % (2 * np.pi)) / (2 * np.pi)
                crop_mapped_slm_phase_0 = utils.crop_image(mapped_slm_phase_0, roi_res, stacked_complex=False)
                # crop_mapped_slm_phase_1 = utils.crop_image(mapped_slm_phase_1, roi_res, stacked_complex=False)

                # writer.add_text(f'val_phs/val_image id', imgs_id[0], total_train_step)
                writer.add_image(f'val_phs/val_phase_0', mapped_slm_phase_0[0,0], total_train_step, dataformats='HW')
                # writer.add_image(f'val_phs/val_phase_1', mapped_slm_phase_1[0,0], total_train_step, dataformats='HW')
                writer.add_histogram(f'val_phs/phase_hist_0', crop_mapped_slm_phase_0[0,0], total_train_step)
                # writer.add_histogram(f'val_phs/phase_hist_1', crop_mapped_slm_phase_1[0,0], total_train_step)
                writer.add_image(f"val_phs/phase_fft_0", real_fourier_transform(crop_mapped_slm_phase_0[0,0]), total_train_step, dataformats='HW')
                # writer.add_image(f"val_phs/phase_fft_1", real_fourier_transform(crop_mapped_slm_phase_1[0,0]), total_train_step, dataformats='HW')
                
                for j in range(len(plane_idx)):
                    writer.add_image(f'val_input_images/plane{j}', masked_imgs[0,j], total_train_step, dataformats='HW')
                    writer.add_images(f'val_output_images/plane{j}', average_scale_factor*outputs_amp[0,j], total_train_step, dataformats='HW')

                writer.flush()
            
            total_val_loss += loss.item()
            total_val_psnr += psnr.item()
            total_val_ssim += ssim.item()
            val_items_count += 1
            
            writer.add_scalar("metrics/inference time", inference_time, total_train_step)
        
        average_val_loss = total_val_loss/val_items_count
        average_val_psnr = total_val_psnr/val_items_count
        average_val_ssim = total_val_ssim/val_items_count
        average_inference_time = total_inference_time/val_items_count
        
        writer.add_scalar("average inference time", average_inference_time, total_train_step)
        if best_val_loss > average_val_loss:
            best_val_loss = average_val_loss
            # save model
            path = f"runs/{run_folder_name}/model/"
            if not os.path.exists(path):
                os.makedirs(path) 
            torch.save(inverse_prop, f"{path}/{run_id}_best_loss.pth")
            writer.add_scalar("ratio/best_scale_factor", average_scale_factor, total_train_step)
            print("best model saved!")
    
    if i % 5 == 0:
        torch.save(inverse_prop, f"{path}/{run_id}_epoch_{i:02d}.pth")
    print(f"epoch_{i:02d} model saved!")
    print(f"Average Val Loss: {average_val_loss}")
    print(f"Average Val PSNR: {average_val_psnr}")
    writer.add_scalar(f"loss/average_val_loss", average_val_loss, total_train_step)
    writer.add_scalar(f"metrics/average_val_psnr", average_val_psnr, total_train_step)
    writer.add_scalar(f"metrics/average_val_ssim", average_val_ssim, total_train_step)
