<h2 align="center"> SpatioTemporal Difference Network for Video Depth Super-Resolution </h2>

## Dependencies

```bash
Python==3.11.5
mmcv-full==1.7.2
torch==2.1.0
numpy==1.23.5 
torchvision==0.16.0
scipy==1.11.3
Pillow==10.0.1
tqdm==4.65.0
scikit-image==0.21.0
```

## Datasets
All datasets can be downloaded from the following link:

[TarTanAir](https://github.com/castacks/tartanair_tools)

[DyDToF](https://github.com/facebookresearch/DVSR/)

[DynamicReplica](https://dynamic-stereo.github.io/)

Additionally, we provide a DyDToF test subset in the ``'dataset'`` folder for quick implementation, with the corresponding index file is ``'data/dydtof_list/school_shot8_subset.txt'``.

## Training

```
cd STDNet
mkdir -p experiment/SRDNet_$scale$/MAE_best

python -m torch.distributed.launch --nproc_per_node 2 train.py --scale 4 --result_root 'experiment/SRDNet_$scale$' --result_root_MAE 'experiment/SRDNet_$scale$/MAE_best'
```

## Testing

```
### TarTanAir dataset
python test_TarTanAir.py --scale 4
### DyDToF dataset
python test_DyDToF.py --scale 4
### DyDToF dataset
python test_DynamicReplica.py --scale 4
```
