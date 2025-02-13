import torch
import os
from load_CNNpropCNN.prop_model import CNNpropCNN, CNNpropCNNprop
from dataclasses import dataclass, field
from typing import List
from load_CNNpropCNN.unet import norm_layer

import sys
sys.path.append(f"..")
sys.path.append(f".")

@dataclass
class PARAMETERS:
    """Class for keeping track of an item in inventory."""
    plane_idxs: List[int] = field(default_factory=lambda: [0])
    image_res: tuple = (1080, 1920)
    num_workers: int = 0
    batch_size: int = 1
    slm_type: str = 'pluto'
    num_downs_slm: int = 8
    num_feats_slm_min: int = 32
    num_feats_slm_max: int = 512
    num_downs_target: int = 5
    num_feats_target_min: int = 8
    num_feats_target_max: int = 128
    prop_dist: float = 0.1
    # wavelength: float = 523.7e-9
    wavelength: float = 5.249e-07
    feature_size: tuple = (8e-6, 8e-6)
    F_aperture: float = 1.0 # no 4f
    prop_dists_from_wrp: List[float] = field(default_factory=lambda: [0.])
    slm_res: tuple = (1080, 1920)
    roi_res: tuple = (880, 1600)
    norm: str = 'instance'
    slm_coord: str = 'rect'
    target_coord: str = 'rect'
    dev: str = 'cuda:0'
    lr: float = 4e-4
    # lr: float = 1e-3
    max_epochs: int = 10000


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_dir = checkpoint['tensor_board_dir']
    
    return model, optimizer, epoch, log_dir


def load_CNNpropCNN(roi_res=None):
    target_dists = [10.0e-2, 10.07e-2, 10.14e-2, 10.21e-2, 10.28e-2, 10.35e-2, 10.42e-2, 10.49e-2]
    # target_dists = [10.00e-2]
    target_dists_str = [f'{dist*1e2:.2f}' for dist in target_dists]
    prop_dist = 10.21e-2
    prop_dists_from_wrp = [target_dist - prop_dist for target_dist in target_dists]
    opt = PARAMETERS(prop_dist=prop_dist, prop_dists_from_wrp=prop_dists_from_wrp,
                    plane_idxs=[0,1,2,3,4,5,6,7])

    # Create the model
    forward_model = CNNpropCNN(opt.prop_dist, opt.wavelength, opt.feature_size,
    # forward_model = CNNpropCNNprop(opt.prop_dist, opt.wavelength, opt.feature_size,
                               prop_type='ASM',
                               F_aperture=opt.F_aperture,
                               prop_dists_from_wrp=opt.prop_dists_from_wrp,
                               linear_conv=True,
                               slm_res=opt.slm_res,
                               roi_res=opt.roi_res if roi_res==None else roi_res,
                               num_downs_slm=opt.num_downs_slm,
                               num_feats_slm_min=opt.num_feats_slm_min,
                               num_feats_slm_max=opt.num_feats_slm_max,
                               num_downs_target=opt.num_downs_target,
                               num_feats_target_min=opt.num_feats_target_min,
                               num_feats_target_max=opt.num_feats_target_max,
                               norm=norm_layer(opt.norm),
                               slm_coord=opt.slm_coord,
                               target_coord=opt.target_coord,
                               ).to(opt.dev)

    optimizer = torch.optim.Adam(forward_model.parameters(), lr=opt.lr)

    ckp = load_checkpoint(model=forward_model, optimizer=optimizer, 
                          load_path="forward_model_best_new.pth")
    forward_model, optimizer, epoch, log_dir = ckp

    forward_model.eval()

    return forward_model