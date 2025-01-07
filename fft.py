import torch
import numpy as np
from torch.fft import fft2, fftshift

def real_fourier_transform(image_tensor, crop_size=(880, 1600)):
    # 将图像转换为灰度图
    if image_tensor.dim() == 3:  # 如果图像有三个通道（RGB）
        gray_image = torch.mean(image_tensor, dim=0)
    else:
        gray_image = image_tensor
    
    # 获取图像的尺寸
    height, width = gray_image.shape
    
    # 计算裁剪区域的中心点
    center_y, center_x = height // 2, width // 2
    
    # 裁剪图像中心区域
    cropped_image = gray_image[
        center_y - crop_size[0] // 2 : center_y + crop_size[0] // 2,
        center_x - crop_size[1] // 2 : center_x + crop_size[1] // 2
    ]
    
    # 对裁剪后的图像执行傅立叶变换
    f = fft2(cropped_image)
    fshift = fftshift(f)
    
    # 计算幅度谱
    magnitude_spectrum = 20 * torch.log10(torch.abs(fshift) + 1e-12)
    
    # 将幅度谱转换为 0-255 范围内的整数
    magnitude_spectrum_normalized = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min()) * 255
    magnitude_spectrum_uint8 = magnitude_spectrum_normalized.to(torch.uint8)
    
    return magnitude_spectrum_uint8
