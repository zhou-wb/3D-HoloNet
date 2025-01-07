import torch
import numpy as np
from scipy.stats import entropy
import imageio.v3 as iio
import torch.nn as nn
import torchvision

def total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return weight*(tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


def dynamic_total_variation_loss(phs, img, weight):
    bs_img, c_img, h_img, w_img = phs.size()
    
    # 计算水平方向的差值
    tv_h = phs[:, :, 1:, :] - phs[:, :, :-1, :]
    # 计算垂直方向的差值
    tv_w = phs[:, :, :, 1:] - phs[:, :, :, :-1]
    
    # 获取对应的像素值作为权重
    weights_h = img[:, :, 1:, :].clone()
    weights_w = img[:, :, :, 1:].clone()
    
    # 应用权重
    weighted_tv_h = torch.pow(tv_h, 2) * weights_h
    weighted_tv_w = torch.pow(tv_w, 2) * weights_w
    
    # 计算加权后的总变差
    tv_h_loss = weighted_tv_h.sum()
    tv_w_loss = weighted_tv_w.sum()
    
    # 归一化
    norm_factor = bs_img * c_img * h_img * w_img
    return weight * (tv_h_loss + tv_w_loss) / norm_factor


def calculate_image_entropy(image_tensor):
    """
    计算图像的信息熵。
    
    Args:
        image_tensor (torch.Tensor): 输入张量，形状为 (B, C, H, W)，其中 B 是批量大小，
                                      C 是通道数,H 是高度,W 是宽度。
    
    Returns:
        torch.Tensor: 一个形状为 (B,) 的张量，包含每个图像的信息熵。
    """
    # 将图像转换为灰度图
    if image_tensor.shape[1] == 3:  # 如果图像有三个通道（RGB）
        gray_image = torch.mean(image_tensor, dim=1, keepdim=True)
    else:
        gray_image = image_tensor
    
    # 计算每个灰度级的出现次数
    # Flatten the tensor and convert it to a one-dimensional histogram
    flat_gray_image = gray_image.reshape(gray_image.size(0), -1)
    histogram_bins = torch.histc(flat_gray_image, bins=256, min=0, max=255)
    
    # 计算每个灰度级出现的概率
    probabilities = histogram_bins / torch.sum(histogram_bins, dim=0, keepdim=True)
    
    # 使用 PyTorch 计算信息熵
    # 注意这里没有提供 qk 参数，意味着默认与均匀分布比较
    entropy_value = -torch.sum(probabilities * torch.log2(probabilities + 1e-12), dim=0)
    
    return entropy_value


# perceptual loss
class PerceptualLoss(torch.nn.modules.loss._Loss):

    def __init__(self, pixel_loss=1.0, l1_loss=False, style_loss=0.0, lambda_feat=1, include_vgg_layers=('1', '2', '3', '4', '5')):
        super(PerceptualLoss, self).__init__(True, True)

        # download pretrained vgg19 if necessary and instantiate it
        vgg19 = torchvision.models.vgg.vgg19(pretrained=True)
        self.vgg_layers = vgg19.features

        # the vgg feature layers we want to use for the perceptual loss
        self.layer_name_mapping = {
        }
        if '1' in include_vgg_layers:
            self.layer_name_mapping['3'] = "conv1_2"
        if '2' in include_vgg_layers:
            self.layer_name_mapping['8'] = "conv2_2"
        if '3' in include_vgg_layers:
            self.layer_name_mapping['13'] = "conv3_2"
        if '4' in include_vgg_layers:
            self.layer_name_mapping['22'] = "conv4_2"
        if '5' in include_vgg_layers:
            self.layer_name_mapping['31'] = "conv5_2"

        # weights for pixel loss and style loss (feature loss assumed 1.0)
        self.pixel_loss = pixel_loss
        self.l1_loss = l1_loss
        self.lambda_feat = lambda_feat
        self.style_loss = style_loss

    def forward(self, input, target):

        lossValue = torch.tensor(0.0).to(input.device)
        l2_loss_func = lambda ipt, tgt: torch.sum(torch.pow(ipt - tgt, 2))  # amplitude to intensity
        l1_loss_func = lambda ipt, tgt: torch.sum(torch.abs(ipt - tgt))  # amplitude to intensity

        # get size
        s = input.size()

        # number of tensors in this mini batch
        num_images = s[0]

        # L2 loss  (L1 originally)
        if self.l1_loss:
            scale = s[1] * s[2] * s[3]
            lossValue += l1_loss_func(input, target) * (2 * self.pixel_loss / scale)
            loss_func = l2_loss_func
        elif self.pixel_loss:
            scale = s[1] * s[2] * s[3]
            lossValue += l2_loss_func(input, target) * (2 * self.pixel_loss / scale)
            loss_func = l2_loss_func

        # stack input and output so we can feed-forward it through vgg19
        x = torch.cat((input, target), 0)

        for name, module in self.vgg_layers._modules.items():

            # run x through current module
            x = module(x)
            s = x.size()

            # scale factor
            scale = s[1] * s[2] * s[3]

            if name in self.layer_name_mapping:
                a, b = torch.split(x, num_images, 0)
                lossValue += self.lambda_feat * loss_func(a, b) / scale

                # Gram matrix for style loss
                if self.style_loss:
                    A = a.reshape(num_images, s[1], -1)
                    B = b.reshape(num_images, s[1], -1).detach()

                    G_A = A @ torch.transpose(A, 1, 2)
                    del A
                    G_B = B @ torch.transpose(B, 1, 2)
                    del B

                    lossValue += loss_func(G_A, G_B) * (self.style_loss / scale)

        return lossValue