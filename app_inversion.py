# this code is largely inspired by https://huggingface.co/spaces/hysts/ControlNet-with-Anything-v4/blob/main/app_scribble_interactive.py
# Thank you, hysts!


import sys
sys.path.append('./src/ControlNetInpaint/')
# functionality based on https://github.com/mikonvergence/ControlNetInpaint

import gradio as gr
import torch
#from torch import autocast // only for GPU

from PIL import Image
import numpy as np
from io import BytesIO
import os

# Usage
# 1. Upload image or fill with white
# 2. Sketch the mask (image->[image,mask]
# 3. Sketch the content of the mask

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

pipe.load_textual_inversion(pretrained_model_name_or_path="/home/urakamiy/repos/diffusers/examples/textual_inversion/textual_inversion_riverplate", weight_name="learned_embeds-steps-1000.bin")

pipe.load_textual_inversion(pretrained_model_name_or_path="/home/urakamiy/repos/diffusers/examples/textual_inversion/textual_inversion_petite-oseille", weight_name="learned_embeds-steps-1000.bin")

if torch.cuda.is_available():
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    pipe.enable_xformers_memory_efficient_attention()

    pipe.to('cuda')

# Functions

css='''
.container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
.image_upload{min-height:500px}
.image_upload [data-testid="image"], .image_upload [data-testid="image"] > div{min-height: 500px}
.image_upload [data-testid="sketch"], .image_upload [data-testid="sketch"] > div{min-height: 500px}
.image_upload .touch-none{display: flex}
#output_image{min-height:500px;max-height=500px;}
'''

def get_guide(image):  
    return hed(image,scribble=True)
    #return canny(image)

def create_demo():
    # Global Storage
    CURRENT_IMAGE={'image': None,
                   'mask': None,
                   'guide': None
                }    
    HEIGHT, WIDTH=512,512
    
    with gr.Blocks(theme=gr.themes.Default(font=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace","monospace"],
                                           primary_hue="lime",
                                           secondary_hue="emerald",
                                           neutral_hue="slate",
                                           ), css=css) as demo:

        gr.Markdown('# Cut and Sketch ✂️▶️✏️')
        with gr.Accordion('Instructions', open=False):
            gr.Markdown('## Cut ✂️')
            gr.Markdown('1. Upload your image below')
            gr.Markdown('2. **Draw the mask** for the region you want changed (Cut ✂️)')
            gr.Markdown('3. Click `Set Mask` when it is ready!')
            gr.Markdown('## Sketch ✏️')
            gr.Markdown('4. Now, you can **sketch a replacement** object! (Sketch ✏️)')
            gr.Markdown('5. (You can also provide a **text prompt** if you want)')
            gr.Markdown('6. 🔮 Click `Generate` when ready! ')             
            example_button=gr.Button(label='example',value='Try example image!').style(full_width=False, size='sm')
            
        with gr.Group():
          with gr.Box():
            with gr.Column():
              with gr.Row() as main_blocks:
                  with gr.Column() as step_1:
                      gr.Markdown('### Mask Input')   
                      image = gr.Image(source='upload',
                                            shape=[HEIGHT,WIDTH],
                                            type='pil',#numpy',
                                            elem_classes="image_upload",
                                       label='Mask Draw (Cut!)',
                                            tool='sketch',
                                            brush_radius=60).style(height=500)
                      input_image=image
                      mask_button = gr.Button(label='Set Mask', value='Set Mask')
                  with gr.Column(visible=False) as step_2:     
                      gr.Markdown('### Sketch Input')         
                      sketch = gr.Image(source='upload',
                                            shape=[HEIGHT,WIDTH],
                                            type='pil',#'numpy',
                                        elem_classes="image_upload",
                                            label='Fill Draw (Sketch!)',
                                            tool='sketch',
                                            brush_radius=10).style(height=500)
                      sketch_image=sketch
                      run_button = gr.Button(label='Generate', value='Generate', variant="primary") 
                      prompt = gr.Textbox(label='Prompt')    
              
                  with gr.Column() as output_step:  
                      gr.Markdown('### Output')   
                      output_image = gr.Gallery(
                                      label="Generated images",
                                      show_label=False,
                                      elem_id="output_image",
                                  ).style(height=500,containter=True)              
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

        with gr.Accordion('More Info', open=False):
            gr.Markdown('This demo was created by Mikolaj Czerkawski [@mikonvergence](https://twitter.com/mikonvergence) based on the 🌱 open-source implementation of [ControlNetInpaint](https://github.com/mikonvergence/ControlNetInpaint) (diffusers-friendly!).')
            gr.Markdown('The tool currently only works with image resolution of 512px.')
            gr.Markdown('💡 To learn more about diffusion with interactive code, check out my open-source ⏩[DiffusionFastForward](https://github.com/mikonvergence/DiffusionFastForward) course. It contains example code, executable notebooks, videos, notes, and a few use cases for training from scratch!')
        
        inputs = [
            sketch_image,
            prompt,
            num_steps,
            text_scale,
            sketch_scale,
            seed
        ]
        
        # STEP 1: Set Mask
        def set_mask(content):
            if content is None:
              gr.Error("You must upload an image first.")
              return {input_image : None,
                      sketch_image : None,
                      step_1: gr.update(visible=True),
                      step_2: gr.update(visible=False)
                  }
                  
            background=np.array(content["image"].convert("RGB").resize((512, 512))) # note: direct numpy seemed buggy
            mask=np.array(content["mask"].convert("RGB").resize((512, 512)))

            if (mask==0).all():
              gr.Error("You must draw a mask for the cut out first.")
              return {input_image : content['image'],
                      sketch_image : None,
                      step_1: gr.update(visible=True),
                      step_2: gr.update(visible=False)
                  }

            mask=1*(mask>0)
            # save vars
            CURRENT_IMAGE['image']=background
            CURRENT_IMAGE['mask']=mask
            
            guide=get_guide(background)
            CURRENT_IMAGE['guide']=np.array(guide)
            guide=255-np.asarray(guide)  
            
            seg_img = guide*(1-mask) + mask*192
            preview = background * (seg_img==255)
            
            vis_image=(preview/2).astype(seg_img.dtype) + seg_img * (seg_img!=255)
            
            return {input_image : content["image"],
                    sketch_image : vis_image,
                    step_1: gr.update(visible=False),
                    step_2: gr.update(visible=True)
                }
        
        # STEP 2: Generate
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
            
            return {output_image : new_image,
                step_1: gr.update(visible=True),
                step_2: gr.update(visible=False)
                }

        def example_fill():

          return Image.open('data/xp-love.jpg')
        
        example_button.click(fn=example_fill, outputs=[input_image])
        mask_button.click(fn=set_mask, inputs=[input_image], outputs=[input_image, sketch_image, step_1,step_2])     
        run_button.click(fn=generate, inputs=inputs, outputs=[output_image, step_1,step_2])
        return demo

if __name__ == '__main__':
    demo = create_demo()
    demo.queue().launch()
