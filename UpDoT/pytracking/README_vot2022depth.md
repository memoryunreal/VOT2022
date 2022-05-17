
## Install the environment
Use the Anaconda
```
conda create -n updot python=3.6
conda activate updot
bash install.sh
```

## Two checkpoints path
You should edit two files as following
line 66: /pytracking/pytracking/parameter/dimp/dimp_ax.py  /path/to/DiMPnet_ep0050_D3.pth.tar
line 63: /pytracking/pytracking/parameter/dimp/dimp_50.py  /path/to/DiMPnet_ep0050_colormap.pth.tar


## Run tracker
[dimp_ax]
label = dimp_ax
protocol = traxpython

command =  run_vot 
paths = /path/to/UpDoT/pytracking/pytracking


## Contact
Zhe Li:liz8@mail.sustech.edu.cn

[[Models]]([DeT](https://github.com/xiaozai/DeT)) 

## Acknowledgments
* Thanks for [VOT](https://www.votchallenge.net/) Library and [DeT](https://github.com/xiaozai/DeT) Library, which helps us to quickly implement our ideas.



