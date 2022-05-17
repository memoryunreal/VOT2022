[[Models and Raw results]](https://drive.google.com/drive/folders/1wyeIs3ytYkmAtTXoVlLMkJ4aSTq5CBHq?usp=sharing) (Google Driver)  [[Models and Raw results]](https://pan.baidu.com/s/1k819gnFMav9t1-8ZhCo74w) (Baidu Driver: hmuv)

## Model path
Put all the downloaded models in /MixFormer/lib/test/networks/mixformer*.pth.tar

## Install the environment
Use the Anaconda
```
conda create -n promix python=3.6
conda activate promix
bash install_pytorch17.sh
```


- VOT2022 RGBD
Before evaluating on VOT2022 RGBD, please install some extra packages following [external/AR/README.md](external/AR/README.md). Also, the VOT toolkit is required to evaluate our tracker. To download and instal VOT toolkit, you can follow this [tutorial](https://www.votchallenge.net/howto/tutorial_python.html). For convenience, you can use our example workspaces of VOT toolkit under ```external/VOT2022RGBD/``` by settings ```trackers.ini```.

create vot workspace, put the tracker.ini in it
vot evalute --workspace /path/to/workspace ProMixTrack
```
vim external/VOT2022RGBD/tracker.ini

* Download the pre-trained Alpha-Refine network  
Download the network for [Alpha-Refine](https://drive.google.com/open?id=1qOQRfaRMbQ2nmgX1NFjoQHfXOAn609QM) 
and put it under the /external/AR/ltr/checkpoints/ltr/ARcm_seg/ARcm_coco_seg_only_mask_384/ARnet_seg_mask_ep0040.pth.tar



## Contact
Zhe Li:liz8@mail.sustech.edu.cn

## Model Zoo and raw results
The trained models and the raw tracking results are provided in the [[Models and Raw results]](https://drive.google.com/drive/folders/1wyeIs3ytYkmAtTXoVlLMkJ4aSTq5CBHq?usp=sharing) (Google Driver) or
[[Models and Raw results]](https://pan.baidu.com/s/1k819gnFMav9t1-8ZhCo74w) (Baidu Driver: hmuv).

## Acknowledgments
* Thanks for [VOT](https://www.votchallenge.net/) Library and [Mixformer](https://github.com/MCG-NJU/MixFormer) Library, which helps us to quickly implement our ideas.



