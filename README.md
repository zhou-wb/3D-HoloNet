# 3D-HoloNet
### [Optical Letters](https://doi.org/10.1364/OL.544816) | [PDF](https://hku.welight.fun/wenbin/assets/pdf/zhou20253d.pdf) | [WeLight Lab @ HKU](https://hku.welight.fun/)

Source codes for the Optical Letters paper titled "3D-HoloNet: Fast, unfiltered, 3D hologram generation with camera-calibrated network learning".

[Wenbin Zhou](https://hku.welight.fun/wenbin),
[Feifan Qu](https://qufeifan.github.io/),
[Xiangyu Meng](https://www.linkedin.com/in/xiangyu-meng-907836302/),
[Zhenyang Li](https://lagrangeli.github.io/),
and [Yifan (Evan) Peng](https://www.eee.hku.hk/~evanpeng/)

## Train the 3D-HoloNet

Download [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) dataset


Set the ```data_path = 'xxx/data/flying3d/'``` in ```train.py``` *Dataset Parameters* section

You may use your own trained forward model *CNNpropCNN* or the ideal propagation model *ASM* for reconstruction. This can be set in the ```train.py``` *Load Networks -- Forward Network* section.
```
python train.py
```

## Inference

Download the checkpoint from [google drive](https://drive.google.com/file/d/1aUQKiLORXeXhXTeLOd5ET0srHOKbdXz3/view?usp=sharing)

Set the ```holo_path['frame_1_with_tv'] = 'xxx/3D-HoloNet_model.pth'``` in ```inference.py``` *Load Models* section

```
python inference.py
```

## Acknowledgements
The codes are built on [neural-holography](https://github.com/computational-imaging/neural-holography) and [neural-3d-holography](https://github.com/computational-imaging/neural-3d-holography). We sincerely appreciate the authors for sharing their codes.
## Contact
If you have any questions, please do not hesitate to contact [zhouwb@connect.hku.hk](zhouwb@connect.hku.hk).