import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from os.path import exists
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import math
import multiprocessing
from multiprocessing import  Process



'''
    args: color file , depth file, depth threshold, blend , colormap style
    return: np.asarray(blend image PIL )
'''
def rgbd_blend(color,depth,depthth=5000,blend=0.05,style="JET"):
    colormap_style = "cv2.COLORMAP_" + style
    if colormap_style in ["cv2.COLORMAP_JET", "cv2.COLORMAP_AUTUMN"]:
         
        style = {
            "cv2.COLORMAP_AUTUMN": 0,
            "cv2.COLORMAP_JET": 2
        }
        depth_threshold = depthth

        # convert depth to colormap
        dp = depth
        dp = cv2.imread(dp, -1)
        try:
            dp[dp>depth_threshold] = depth_threshold # ignore some large values,
        except:
            dp = dp
        dp = cv2.normalize(dp, None, 0, 255, cv2.NORM_MINMAX)
        dp = cv2.applyColorMap(np.uint8(dp), style[colormap_style])
        # dp = cv2.applyColorMap(np.uint8(dp), cv2.COLORMAP_JET)

        # cv2.imwrite(os.path.join(colormap_path, filename),dp)

        # blend colormap and color
        color_file = color
        colorf = Image.open(color_file)
        colorf = colorf.convert('RGBA')

        # colormapf = Image.open(colormapfile)
        colormapf = Image.fromarray(cv2.cvtColor(dp,cv2.COLOR_BGR2RGB))

        colormapf = colormapf.convert('RGBA')

        blend_img = Image.blend(colorf, colormapf, blend)
    elif colormap_style == "cv2.COLORMAP_Gray":
        depth_threshold = depthth

        # convert depth to colormap
        dp = depth
        dp = cv2.imread(dp, -1)
        try:
            dp[dp>depth_threshold] = depth_threshold # ignore some large values,
        except:
            dp = dp
        dp = cv2.normalize(dp, None, 0, 255, cv2.NORM_MINMAX)

        # blend colormap and color
        color_file = color
        colorf = Image.open(color_file)
        colorf = colorf.convert('RGBA')

        # colormapf = Image.open(colormapfile)
        # colormapf = Image.fromarray(cv2.cvtColor(dp,cv2.COLOR_BGR2RGB))
        colormapf = Image.fromarray(dp)

        colormapf = colormapf.convert('RGBA')

        blend_img = Image.blend(colorf, colormapf, blend) 
    #img.show()
    
    # blend_img.save(save)
    
    # dir rename: color -> oldcolor newcolor -> color
    result = cv2.cvtColor(np.asarray(blend_img), cv2.COLOR_RGB2BGR)
    return result



        