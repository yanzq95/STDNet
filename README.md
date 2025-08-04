<h2 align="center"> SpatioTemporal Difference Network for Video Depth Super-Resolution </h2>

<p align="center"><a href="https://scholar.google.com/citations?user=VogTuQkAAAAJ&hl=zh-CN">Zhengxue Wang</a><sup>1</sup>, 
<a href="https://scholar.google.com.hk/citations?user=VoIgY38AAAAJ&hl=zh-CN">Yuan Wu</a><sup>1</sup>,
<a href="https://implus.github.io/">Xiang Li</a><sup>2</sup>,
  <a href="https://yanzq95.github.io/">Zhiqiang Yan✉</a><sup>3</sup>, 
<a href="https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN">Jian Yang✉</a><sup>1</sup>  <!--&Dagger;-->
</p>

<p align="center">
  <sup>✉</sup>Corresponding author&nbsp;&nbsp;&nbsp;<br>
  <sup>1</sup>Nanjing University of Science and Technology&nbsp;&nbsp;&nbsp;
  <br>
  <sup>2</sup>Nankai University&nbsp;&nbsp;&nbsp;
  <sup>3</sup>National University of Singapore&nbsp;&nbsp;&nbsp;
</p>


<p align="center">
<img src="Figs/Pipeline.PNG", width="850"/>
</p>

Overview of STDNet. Given $\boldsymbol D_{LR}$, we first predict its spatial difference representation $\boldsymbol \sigma$. Then, $\boldsymbol D_{LR}$, $\boldsymbol I$, and $\boldsymbol \sigma$ are jointly fed into the spatial difference to enhance non-smooth regions, producing $\boldsymbol F_{sd}$. Next, we estimate the temporal difference representations for consecutive frames and cross frames, generating $\boldsymbol \varphi$ and $\widehat{\boldsymbol \varphi}$. These difference representations are used to propagate adjacent RGB and depth frames to the current depth frame, generating HR depth video $\boldsymbol D_{HR}$. Finally, a degradation regularization takes $\boldsymbol D_{HR}$, $\boldsymbol D_{GT}$, $\boldsymbol \sigma$, $\boldsymbol \varphi$, and $\widehat{\boldsymbol \varphi}$ as inputs to optimize the learning of spatiotemporal difference representations.


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

## Models
All pretrained models can be found <a href="https://drive.google.com/drive/folders/14MsOiHI2xIJ9w07hI-xrsX_1fHvoAAhq?usp=sharing">here</a>.

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

## Experiments

### Quantitative comparison

<p align="center">
<img src="Figs/qc.png", width="500"/>
<br>
Quantitative comparisons between our STDNet and previous state-of-the-art methods on TarTanAir dataset.
</p>



### Visual comparison

<p align="center">
<img src="Figs/visual.png", width="900"/>
<br>
Visual results on the DyDToF and DynamicReplica datasets ($\times8$).
</p>
