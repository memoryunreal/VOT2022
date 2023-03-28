# VOT2022
# ProMixTrack
Configuration
```
Host: 10.20.111.5
Path: /home/chenhongjun/lz/vot2022
Run docker: /home/chenhongjun/lz/docker/vot2022.sh 
Dockerimage: watchtowerss/vot2022:latest
Workspace: /home/chenhongjun/lz/vot2022/vot2022/workspace/

Host: 10.20.4.112
Path: /ssd3/lz/VOT2022
Run docker: /ssd3/lz/VOT2022/vot2022.sh
Dockerimage: watchtowerss/vot2022:mixformer
Workspace: /ssd3/lz/VOT2022/vot2022/workspace
Submit: /ssd3/lz/VOT2022/submit
```

## Install the environment
Use the Anaconda
```
conda create -n promix python=3.6
conda activate promix
bash install_pytorch17.sh
```
## Model path
* Models and results for mixformer 
[[Models and Raw results]](https://drive.google.com/drive/folders/1wyeIs3ytYkmAtTXoVlLMkJ4aSTq5CBHq?usp=sharing) (Google Driver)  [[Models and Raw results]](https://pan.baidu.com/s/1k819gnFMav9t1-8ZhCo74w) (Baidu Driver: hmuv) 
* Download the pre-trained  [Alpha-Refine](https://drive.google.com/open?id=1qOQRfaRMbQ2nmgX1NFjoQHfXOAn609QM)  network   (vot only)

```
Download models from Google Driver or Baidu Driver 
Models path: /Project/to/MixFormer/lib/test/networks/mixformer*.pth.tar

# Download Alpha-Refine models (vot only)
Model path: /external/AR/ltr/checkpoints/ltr/ARcm_seg/ARcm_coco_seg_only_mask_384/ARnet_seg_mask_ep0040.pth.tar
```

## Experiments on VOT2022 RGBD
Before evaluating on VOT2022 RGBD, please install some extra packages following [external/AR/README.md](external/AR/README.md). Also, the VOT toolkit is required to evaluate our tracker. To download and instal VOT toolkit, you can follow this [tutorial](https://www.votchallenge.net/howto/tutorial_python.html). For convenience, you can use our example workspaces of VOT toolkit under ```external/VOT2022RGBD/``` by settings ```trackers.ini```.

```
# Check the tracker.ini
vim external/VOT2022RGBD/tracker.ini

# Create vot workspace dir
vot initialize vot2022/rgbd --workspace ./

# Evaluate
vot evaluate --workspace ./ ProMixTrack

# Analysis
vot analysis --workspace ./ ProMixTrack
```



## Contact
Zhe Li:liz2022@mail.sustech.edu.cn

## Model Zoo and raw results
The trained models and the raw tracking results are provided in the [[Models and Raw results]](https://drive.google.com/drive/folders/1wyeIs3ytYkmAtTXoVlLMkJ4aSTq5CBHq?usp=sharing) (Google Driver) or
[[Models and Raw results]](https://pan.baidu.com/s/1k819gnFMav9t1-8ZhCo74w) (Baidu Driver: hmuv).

## Acknowledgments
* Thanks for [VOT](https://www.votchallenge.net/) Library and [Mixformer](https://github.com/MCG-NJU/MixFormer) Library, which helps us to quickly implement our ideas.
