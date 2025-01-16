import torch
import argparse
import numpy as np
import torch.distributed as dist
import os
from PIL import Image
from tqdm.auto import tqdm
import json
from dataclasses import dataclass


from relighting.inpainter import BallInpainter

from relighting.mask_utils import MaskGenerator
from relighting.ball_processor import (
    get_ideal_normal_ball,
    crop_ball
)
from relighting.dataset import GeneralLoader
from relighting.utils import name2hash
from relighting.environment_map_projector import EnvironmentMapProjector
from relighting.environment_map_projector import EnvironmentMapProjector
from relighting.exposure_bracketer import ExposureBracketer
from relighting.chrome_ball_generator import ChromeBallGenerator
import relighting.dist_utils as dist_util
import time


class EnvironmentMapGenerator:
    def __init__(self):
        self.chromeBallGenerator = ChromeBallGenerator()
        self.environmentMapProjector = EnvironmentMapProjector()
        self.exposureBracketer = ExposureBracketer()

    def load_model(self):
        self.chromeBallGenerator.load_model()

    def __call__(self, image: Image.Image, seed: int) -> np.ndarray:
        cropped_spheres = self.chromeBallGenerator.generate_spheres(image, seed)
        env_maps = {}
        for ev, im in cropped_spheres.items():
            # im.save(f"output/{ev}.png")
            env_maps[ev] = self.environmentMapProjector(im) / 255
            # Image.fromarray((env_maps[ev] * 255).astype(np.uint8)).save(f"output/bed_ev-{int(ev*10)}_map.png")
        hdr_env_map = self.exposureBracketer(env_maps)
        # Env maps are coming out flipped... dont ask me why
        hdr_env_map = np.flipud(hdr_env_map)
        return hdr_env_map
