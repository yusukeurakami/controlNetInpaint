# this code is largely inspired by https://huggingface.co/spaces/hysts/ControlNet-with-Anything-v4/blob/main/app_scribble_interactive.py
# Thank you, hysts!

import sys
sys.path.append('./src/ControlNetInpaint/')
# functionality based on https://github.com/mikonvergence/ControlNetInpaint

import gradio as gr
#import torch
#from torch import autocast // only for GPU

from PIL import Image
import numpy as np
from io import BytesIO
import os

# Usage
# 1. Upload image or fill with white
# 2. Sketch the mask (image->[image,mask]
# 3. Sketch the content of the mask

# Global Storage
CURRENT_IMAGE={'image' : None,
               'mask' : None,
               'guide' : None
            }

HEIGHT,WIDTH=512,512

## SETUP PIPE

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image
from controlnet_aux import HEDdetector

hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-scribble", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

if torch.cuda.is_available():
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    pipe.enable_xformers_memory_efficient_attention()

    pipe.to('cuda')

# Functions

def get_guide(image):  
  return hed(image,scribble=True)

def generate(image,
             prompt,
             num_steps,
             text_scale,
             sketch_scale,
             seed):

  sketch=(255*(image['mask'][...,:3]>0)).astype(CURRENT_IMAGE['image'].dtype)
  mask=CURRENT_IMAGE['mask']
  
  CURRENT_IMAGE['guide']=(CURRENT_IMAGE['guide']*(mask==0) + sketch*(mask!=0)).astype(CURRENT_IMAGE['image'].dtype)

  mask_img=255*CURRENT_IMAGE['mask'].astype(CURRENT_IMAGE['image'].dtype)

  new_image = pipe(
      prompt,
      num_inference_steps=num_steps,
      guidance_scale=text_scale,
      generator=torch.manual_seed(seed),
      image=Image.fromarray(CURRENT_IMAGE['image']),
      control_image=Image.fromarray(CURRENT_IMAGE['guide']),
      controlnet_conditioning_scale=sketch_scale,
      mask_image=Image.fromarray(mask_img)
  ).images#[0]

  return new_image

def create_demo(max_images=12, default_num_images=3):
    with gr.Blocks(theme=gr.themes.Default(font=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace","monospace"])) as demo:
        gr.Markdown('## Cut and Sketch ✂️▶️✏️')
    
        with gr.Column() as step_1:
            gr.Markdown('**Start Here**')
            gr.Markdown('1. Upload your image below')
            gr.Markdown('2. **Draw the mask** for the region you want changed (Cut ✂️)')
            input_image = gr.Image(source='upload',
                                    shape=[HEIGHT,WIDTH],
                                    type='numpy',
                                  label='Mask Draw (Cut!)',
                                    tool='sketch',
                                    brush_radius=80)
            gr.Markdown('3. Click `Set Mask` when it is ready!')
            mask_button = gr.Button(label='Set Mask', value='Set Mask')
        with gr.Column(visible=False) as step_2:
            gr.Markdown('4. Now, you can **sketch a replacement** object! (Sketch ✏️)')
            sketch_image = gr.Image(source='upload',
                                          shape=[HEIGHT,WIDTH],
                                          type='numpy',
                                        label='Fill Draw (Sketch!)',
                                          tool='sketch',
                                          brush_radius=20)
            gr.Markdown('5. (You can also provide a **text prompt** if you want)')
            prompt = gr.Textbox(label='Prompt')   
            gr.Markdown('6. 🔮 Click `Generate` when ready! ')   
            run_button = gr.Button(label='Generate', value='Generate')         
            output_image = gr.Gallery(
                              label="Generated images",
                              visible=False,
                              show_label=False,
                              elem_id="gallery",
                          )#.style(grid=(1, 2))
        
            with gr.Accordion('Advanced options', open=False):
                num_steps = gr.Slider(label='Steps',
                                      minimum=1,
                                      maximum=100,
                                      value=20,
                                      step=1)
                text_scale = gr.Slider(label='Text Guidance Scale',
                                            minimum=0.1,
                                            maximum=30.0,
                                            value=7.5,
                                            step=0.1)
                seed = gr.Slider(label='Seed',
                                  minimum=-1,
                                  maximum=2147483647,
                                  step=1,
                                  randomize=True)  
            
                sketch_scale = gr.Slider(label='Sketch Guidance Scale',
                                            minimum=0.0,
                                            maximum=1.0,
                                            value=1.0,
                                            step=0.05)                  
          
        inputs = [
          sketch_image,
          prompt,
          num_steps,
          text_scale,
          sketch_scale,
          seed
        ]
        
        def set_mask(image):
            img=image['image'][...,:3]
            mask=1*(image['mask'][...,:3]>0)
            # save vars
            CURRENT_IMAGE['image']=img
            CURRENT_IMAGE['mask']=mask
            
            guide=get_guide(img)
            CURRENT_IMAGE['guide']=np.array(guide)
            guide=255-np.asarray(guide)  
            
            seg_img = guide*(1-mask) + mask*192
            preview = img * (seg_img==255)
            
            vis_image=(preview/2).astype(seg_img.dtype) + seg_img * (seg_img!=255)
            
            return {input_image : image['image'], sketch_image : vis_image, step_2: gr.update(visible=True)}
        
        mask_button.click(fn=set_mask, inputs=[input_image], outputs=[input_image, sketch_image,step_2])     
        run_button.click(fn=generate, inputs=inputs, outputs=output_image)
    return demo

if __name__ == '__main__':
    demo = create_demo()
    demo.queue().launch()