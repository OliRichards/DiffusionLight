import numpy as np
from PIL import Image
import skimage
import time
import torch
import argparse 
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import os

try:
    import ezexr
except:
    pass


class EnvironmentMapProjector:
    def create_envmap_grid(self, size: int) -> np.ndarray:
        """
        BLENDER CONVENSION
        Create the grid of environment map that contain the position in sperical coordinate
        Top left is (0,0) and bottom right is (pi/2, 2pi)
        """    
        
        theta = torch.linspace(0, np.pi * 2, size * 2)
        phi = torch.linspace(0, np.pi, size)
        
        #use indexing 'xy' torch match vision's homework 3
        theta, phi = torch.meshgrid(theta, phi ,indexing='xy') 
        
        
        theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
        theta_phi = theta_phi.numpy()
        return theta_phi

    def get_normal_vector(self, incoming_vector: np.ndarray, reflect_vector: np.ndarray):
        """
        BLENDER CONVENSION
        incoming_vector: the vector from the point to the camera
        reflect_vector: the vector from the point to the light source
        """
        #N = 2(R ⋅ I)R - I
        N = (incoming_vector + reflect_vector) / np.linalg.norm(incoming_vector + reflect_vector, axis=-1, keepdims=True)
        return N

    def get_cartesian_from_spherical(self, theta: np.array, phi: np.array, r = 1.0):
        """
        BLENDER CONVENSION
        theta: vertical angle
        phi: horizontal angle
        r: radius
        """
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.concatenate([x[...,None],y[...,None],z[...,None]], axis=-1)
        
        
    def process_image(self, ball_image:np.ndarray, envmap_height: int = 256, scale: float = 4) -> np.ndarray:
        I = np.array([1,0, 0])

        # compute  normal map that create from reflect vector
        env_grid = self.create_envmap_grid(envmap_height * scale)   
        reflect_vec = self.get_cartesian_from_spherical(env_grid[...,1], env_grid[...,0])
        normal = self.get_normal_vector(I[None,None], reflect_vec)
        
        # turn from normal map to position to lookup [Range: 0,1]
        pos = (normal + 1.0) / 2
        pos  = 1.0 - pos
        pos = pos[...,1:]
        
        env_map = None
        
        # using pytorch method for bilinear interpolation
        with torch.no_grad():
            # convert position to pytorch grid look up
            grid = torch.from_numpy(pos)[None].float()
            grid = grid * 2 - 1 # convert to range [-1,1]

            # convert ball to support pytorch
            ball_image = torch.from_numpy(ball_image[None]).float()
            ball_image = ball_image.permute(0,3,1,2) # [1,3,H,W]
            
            env_map = torch.nn.functional.grid_sample(ball_image, grid, mode='bilinear', padding_mode='border', align_corners=True)
            env_map = env_map[0].permute(1,2,0).numpy()
                    
        env_map_default = skimage.transform.resize(env_map, (envmap_height, envmap_height*2), anti_aliasing=True)
        return env_map_default


    def __call__(self, ball: Image.Image) -> np.ndarray:
        return self.process_image(np.array(ball))