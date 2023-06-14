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
    "fusing/stable-diffusion-v1-5-controlnet-canny", torch_dtype=torch.float16
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

data_folder = './data'
output_folder = './results'

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
    
    #CURRENT_IMAGE['guide']=(CURRENT_IMAGE['guide']*(mask==0) + sketch*(mask!=0)).astype(CURRENT_IMAGE['image'].dtype)
    CURRENT_IMAGE['guide']=sketch*(mask!=0).astype(CURRENT_IMAGE['image'].dtype)

    mask_img=255*CURRENT_IMAGE['mask'].astype(CURRENT_IMAGE['image'].dtype)

    cv2.imwrite(os.path.join(output_folder, "image.png"), cv2.cvtColor(CURRENT_IMAGE['image'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, "sketch.png"), sketch)
    cv2.imwrite(os.path.join(output_folder, "guide.png"), CURRENT_IMAGE['guide'])
    cv2.imwrite(os.path.join(output_folder, "mask_img.png"), mask_img)

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

def main(args):
    contents = {}
    image = cv2.cvtColor(cv2.imread(os.path.join(data_folder, "target.png")), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (HEIGHT, WIDTH))
    CURRENT_IMAGE['image'] = image
    mask = cv2.imread(os.path.join(data_folder, 'mask.png'), cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (HEIGHT, WIDTH))
    CURRENT_IMAGE['mask'] = mask/255 if 255 in np.unique(mask) else mask
    ref_image = cv2.cvtColor(cv2.imread(os.path.join(data_folder, "ref.png")), cv2.COLOR_BGR2RGB)
    ref_image = cv2.resize(ref_image, (HEIGHT, WIDTH))
    CURRENT_IMAGE['guide'] = np.array(get_guide(image))
    contents['mask'] = get_guide(ref_image)

    new_image = generate(contents, args.prompt, num_steps=args.num_steps, text_scale=args.text_scale, sketch_scale=args.sketch_scale, seed=0)
    new_image = np.array(new_image[0])
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, "new_image.png"), new_image)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="photo of <petite-oseille>")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--text_scale", type=float, default=7.5)
    parser.add_argument("--sketch_scale", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
