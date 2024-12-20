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
import relighting.dist_utils as dist_util
import time

# cross import from inpaint_multi-illum.py
from relighting.argument import (
    SD_MODELS, 
    CONTROLNET_MODELS,
    VAE_MODELS
)

@dataclass
class RuntimeConfig:
    lora_scale: float = 0.75
    guidance_scale: float = 5.0
    ball_dilate: float = 20
    ball_size: float = 256
    prompt: str = "a perfect mirrored reflective chrome ball sphere"
    prompt_dark: str = "a perfect black dark mirrored reflective chrome ball sphere"
    negative_prompt: str = "matte, diffuse, flat, dull"
    denoising_step: int = 30
    control_scale: float = 0.5
    agg_mode: str = "median"
    num_iteration: int = 2
    ball_per_iteration: int = 30
    # ev: tuple[float] = (0, -2.5, -5)
    ev: tuple[float] = (0, -5)
    max_negative_ev: float = -5
    lora_scale: float = 0.75

@dataclass
class PipelineConfig:
    model_option: str = "sdxl" # "sdxl" "sdxl_turbo" "sdxl_fast"
    #lora_path: str = "models/ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500"
    lora_path: str = "DiffusionLight/DiffusionLight"
    img_height: int = 1024
    img_width: int = 1024



def get_ball_location(image_data, args):
    if 'boundary' in image_data:
        # support predefined boundary if need
        x = image_data["boundary"]["x"]
        y = image_data["boundary"]["y"]
        r = image_data["boundary"]["size"]
        
        # support ball dilation
        half_dilate = args.ball_dilate // 2

        # check if not left out-of-bound
        if x - half_dilate < 0: x += half_dilate
        if y - half_dilate < 0: y += half_dilate

        # check if not right out-of-bound
        if x + r + half_dilate > args.img_width: x -= half_dilate
        if y + r + half_dilate > args.img_height: y -= half_dilate   
            
    else:
        # we use top-left corner notation
        x, y, r = ((args.img_width // 2) - (args.ball_size // 2), (args.img_height // 2) - (args.ball_size // 2), args.ball_size)
    return x, y, r

class ChromeBallGenerator:
    def __init__(self, pipelineConfig: PipelineConfig = PipelineConfig(), runtimeConfig: RuntimeConfig = RuntimeConfig()):
        self.device = dist_util.dev()
        self.torch_dtype = torch.float16
        self.pipelineConfig = pipelineConfig
        self.runtimeConfig = runtimeConfig

    def interpolate_embedding(self):
        print("interpolate embedding...")

        # get list of all EVs
        ev_list = [float(x) for x in self.runtimeConfig.ev]
        interpolants = [ev / self.runtimeConfig.max_negative_ev for ev in ev_list]

        print("EV : ", ev_list)
        print("EV : ", interpolants)

        # calculate prompt embeddings
        prompt_normal = self.runtimeConfig.prompt
        prompt_dark = self.runtimeConfig.prompt_dark
        prompt_embeds_normal, _, pooled_prompt_embeds_normal, _ = self.pipe.pipeline.encode_prompt(prompt_normal)
        prompt_embeds_dark, _, pooled_prompt_embeds_dark, _ = self.pipe.pipeline.encode_prompt(prompt_dark)

        # interpolate embeddings
        interpolate_embeds = []
        for t in interpolants:
            int_prompt_embeds = prompt_embeds_normal + t * (prompt_embeds_dark - prompt_embeds_normal)
            int_pooled_prompt_embeds = pooled_prompt_embeds_normal + t * (pooled_prompt_embeds_dark - pooled_prompt_embeds_normal)

            interpolate_embeds.append((int_prompt_embeds, int_pooled_prompt_embeds))

        return dict(zip(ev_list, interpolate_embeds))


    def load_model(self):
        # create controlnet pipeline 
        model_option = self.pipelineConfig.model_option
        if model_option in ["sdxl", "sdxl_fast", "sdxl_turbo"]:
            model, controlnet = SD_MODELS[model_option], CONTROLNET_MODELS[model_option]
            self.pipe = BallInpainter.from_sdxl(
                model=model, 
                controlnet=controlnet, 
                device=self.device,
                torch_dtype = self.torch_dtype,
                offload = False
            )
        if model_option in ["sdxl_turbo"]:
            # Guidance scale is not supported in sdxl_turbo
            self.runtimeConfig.guidance_scale = 0.0
        
        if self.runtimeConfig.lora_scale > 0 and self.pipelineConfig.lora_path is None:
            raise ValueError("lora scale is not 0 but lora path is not set")
        
        if (self.pipelineConfig.lora_path is not None):
            print(f"using lora path {self.pipelineConfig.lora_path}")
            print(f"using lora scale {self.runtimeConfig.lora_scale}")
            self.pipe.pipeline.load_lora_weights(self.pipelineConfig.lora_path)
            self.pipe.pipeline.fuse_lora(lora_scale=self.runtimeConfig.lora_scale) # fuse lora weight w' = w + \alpha \Delta w
            enabled_lora = True
        else:
            enabled_lora = False

        if True:
            try:
                print("compiling unet model")
                start_time = time.time()
                self.pipe.pipeline.unet = torch.compile(self.pipe.pipeline.unet, mode="reduce-overhead", fullgraph=True)
                print("Model compilation time: ", time.time() - start_time)
            except:
                pass
                    
        # default height for sdxl is 1024, if not set, we set default height.
        if self.pipelineConfig.model_option == "sdxl" and self.pipelineConfig.img_height == 0 and self.pipelineConfig.img_width == 0:
            self.pipelineConfig.img_height = 1024
            self.pipelineConfig.img_width = 1024
        
        print(f"Loaded model!")

    def generate_spheres(self, image: Image.Image, seed: int) -> dict[float, Image.Image]:
        # so, we need ball_dilate >= 16 (2*vae_scale_factor) to make our mask shape = (272, 272)
        assert self.runtimeConfig.ball_dilate % 2 == 0 # ball dilation should be symmetric

        # TODO @(oliri): It seems dodgy to resize without maintaining aspect ratio. But this is what they were doing...
        image.thumbnail((self.pipelineConfig.img_width, self.pipelineConfig.img_height))
            
        # interpolate embedding
        embedding_dict = self.interpolate_embedding()
        
        # prepare mask and normal ball
        mask_generator = MaskGenerator()
        normal_ball, mask_ball = get_ideal_normal_ball(size=self.runtimeConfig.ball_size + self.runtimeConfig.ball_dilate)
        _, mask_ball_for_crop = get_ideal_normal_ball(size=self.runtimeConfig.ball_size)
        
        images = {}
        for ev, (prompt_embeds, pooled_prompt_embeds) in embedding_dict.items():
            # create output file name (we always use png to prevent quality loss)
            ev_str = str(ev).replace(".", "") if ev != 0 else "-00"
            #outname = os.path.basename(image_path).split(".")[0] + f"_ev{ev_str}"

            # we use top-left corner notation (which is different from aj.aek's center point notation)
            x, y, r = ((image.width // 2) - (self.runtimeConfig.ball_size // 2), (image.height // 2) - (self.runtimeConfig.ball_size // 2), self.runtimeConfig.ball_size)
            
            # create inpaint mask
            mask = mask_generator.generate_single(
                image, mask_ball, 
                x - (self.runtimeConfig.ball_dilate // 2),
                y - (self.runtimeConfig.ball_dilate // 2),
                r + self.runtimeConfig.ball_dilate
            )
                
            start_time = time.time()
            # skip if file exist, useful for resuming
            generator = torch.Generator().manual_seed(seed)
            kwargs = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                'negative_prompt': self.runtimeConfig.negative_prompt,
                'num_inference_steps': self.runtimeConfig.denoising_step,
                'generator': generator,
                'image': image,
                'mask_image': mask,
                'strength': 1.0,
                'current_seed': seed, # we still need seed in the pipeline!
                'controlnet_conditioning_scale': self.runtimeConfig.control_scale,
                'height': self.pipelineConfig.img_height,
                'width': self.pipelineConfig.img_width,
                'normal_ball': normal_ball,
                'mask_ball': mask_ball,
                'x': x,
                'y': y,
                'r': r,
                'guidance_scale': self.runtimeConfig.guidance_scale,
            }
            
            kwargs["cross_attention_kwargs"] = {"scale": self.runtimeConfig.lora_scale}
            
            output_image = self.pipe.inpaint(**kwargs).images[0]
            print("Ran model!")
                
                
            square_image = output_image.crop((x, y, x+r, y+r))
            images[ev] = square_image

            # return the most recent control_image for sanity check
            #control_image = self.pipe.get_cache_control_image()
            #if control_image is not None:
            #    control_image.save(os.path.join(control_output_dir, outpng))
            
            # save image 
            #output_image.save(os.path.join(raw_output_dir, outpng))
            #square_image.save(os.path.join(square_output_dir, outpng))
        return images
