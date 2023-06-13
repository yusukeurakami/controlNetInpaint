#!/local/env/bin python

# this code is largely inspired by https://huggingface.co/spaces/hysts/ControlNet-with-Anything-v4/blob/main/app_scribble_interactive.py
# Thank you, hysts!


import sys

sys.path.append('./src/ControlNetInpaint/')
# functionality based on https://github.com/mikonvergence/ControlNetInpaint
import cv2

import gradio as gr
import torch
#from torch import autocast // only for GPU

from PIL import Image
import numpy as np
from io import BytesIO
import os

## SETUP PIPE
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import HEDdetector, CannyDetector

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
canny = CannyDetector()

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-scribble", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

from textural_inversion_config import textural_inversion_file_dict
for path, weight in textural_inversion_file_dict.items():
    pipe.load_textual_inversion(pretrained_model_name_or_path=path, weight_name=weight)

if torch.cuda.is_available():
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    pipe.enable_xformers_memory_efficient_attention()

    pipe.to('cuda')

CURRENT_IMAGE={'image': None,
               'mask': None,
               'guide': None
            }    
HEIGHT, WIDTH=512,512

def get_guide(image, guide_type="canny"):  
    if guide_type == "scribble":
        return hed(image,scribble=True)
    elif guide_type == "canny":
        return Image.fromarray(np.repeat(canny(image)[..., np.newaxis], 3, axis=2))    

def generate(content,
     prompt,
     num_steps,
     text_scale,
     sketch_scale,
     seed):
    sketch=np.array(content["mask"].convert("RGB").resize((512, 512)))            
    sketch=(255*(sketch>0)).astype(CURRENT_IMAGE['image'].dtype)
    mask=CURRENT_IMAGE['mask']
    
    CURRENT_IMAGE['guide']=(CURRENT_IMAGE['guide']*(mask==0) + sketch*(mask!=0)).astype(CURRENT_IMAGE['image'].dtype)
    breakpoint()

    mask_img=255*CURRENT_IMAGE['mask'].astype(CURRENT_IMAGE['image'].dtype)

    new_image = pipe(
         prompt,
         num_inference_steps=num_steps,
         guidance_scale=text_scale,
         generator=torch.manual_seed(seed),
         image=Image.fromarray(CURRENT_IMAGE['image']),
         control_image=Image.fromarray(CURRENT_IMAGE['guide']),
         controlnet_conditioning_scale=float(sketch_scale),
         mask_image=Image.fromarray(mask_img)
        ).images#[0]
    return new_image

def main():
    contents = {}
    image = cv2.cvtColor(cv2.imread("data/riverplate_target.png"), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (HEIGHT, WIDTH))
    CURRENT_IMAGE['image'] = image
    mask = cv2.imread('data/mask.png', cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (HEIGHT, WIDTH))
    CURRENT_IMAGE['mask'] = mask
    ref_image = cv2.cvtColor(cv2.imread("data/riverplate_ref.png"), cv2.COLOR_BGR2RGB)
    ref_image = cv2.resize(image, (HEIGHT, WIDTH))
    CURRENT_IMAGE['guide'] = np.array(get_guide(ref_image))
    contents['mask'] = Image.fromarray(CURRENT_IMAGE['guide'])
    

    guide=CURRENT_IMAGE['guide']  
    seg_img = guide*(1-mask) + mask*192
    preview = image * (seg_img==255)
    vis_image=(preview/2).astype(seg_img.dtype) + seg_img * (seg_img!=255)
    cv2.imwrite('preview.png', vis_image)

    prompt = "green <petite-oseille>"
    new_image = generate(contents, prompt, num_steps=50, text_scale=7.5, sketch_scale=1.0, seed=0)
    new_image = np.array(new_image[0])
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("new_image.png", new_image)

if __name__ == "__main__":
    main()
