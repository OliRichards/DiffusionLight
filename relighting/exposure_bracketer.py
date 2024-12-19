# covnert exposure bracket to HDR output
import argparse 
import os 
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import skimage
import ezexr
from relighting.tonemapper import TonemapHDR

class ExposureBracketer:
    def __call__(self, exposures: dict[float, np.ndarray]) -> np.ndarray:
        scaler = np.array([0.212671, 0.715160, 0.072169])
        # ev value for each file
        # evs = [e for e in sorted(exposures.keys(), reverse = True)]
        evs = [e for e in sorted(exposures.keys(), reverse=True)]

        # inital first image
        image0 = exposures[evs[0]]
        image0 = skimage.img_as_float(image0)
        image0_linear = np.power(image0, 2.4)

        # read luminace for every image 
        luminances = []
        for ev in evs:
            # load image
            image = exposures[ev]
            image = skimage.img_as_float(image)
            
            # apply gama correction
            linear_img = np.power(image, 2.4)
            
            # convert the brighness
            linear_img *= 1 / (2 ** (ev))
            
            # compute luminace
            lumi = linear_img @ scaler
            luminances.append(lumi)
            
        # start from darkest image
        out_luminace = luminances[len(evs) - 1]
        for i in range(len(evs) - 1, 0, -1):
            # compute mask
            maxval = 1 / (2 ** evs[i-1])
            p1 = np.clip((luminances[i-1] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
            p2 = out_luminace > luminances[i-1]
            mask = (p1 * p2).astype(np.float32)
            out_luminace = luminances[i-1] * (1-mask) + out_luminace * mask
            
        hdr_rgb = image0_linear * (out_luminace / (luminances[0] + 1e-10))[:, :, np.newaxis]
        
        # tone map for visualization    
        # hdr2ldr = TonemapHDR(gamma=self.gamma, percentile=99, max_mapping=0.9)

        return hdr_rgb