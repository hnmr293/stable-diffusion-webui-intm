from collections import defaultdict
from typing import TypeAlias, Callable
import numpy as np
from PIL import Image
import gradio as gr
import torch
from torch import nn, Tensor

from modules import scripts, shared
from modules.processing import StableDiffusionProcessing, process_images, state, decode_first_stage
from modules.sd_samplers import KDiffusionSampler, VanillaStableDiffusionSampler
from scripts.intmlib import putils

# image_index -> step -> Tensor
TensorResult: TypeAlias = defaultdict[int, list[Tensor]]

# image_index -> step -> Image
ImageResult: TypeAlias = defaultdict[int, list[Image.Image]]

class Hooker:
    
    def __init__(self, target, method: str, fn: Callable):
        org = getattr(target, method)
        self.target = target
        self.method = method
        self.org = org
        print(f"[Intm] Hooking {target.__name__}.{method}")
        setattr(target, method, self.create_hook(fn))
    
    def remove(self):
        print(f"[Intm] Unhooking {self.target.__name__}.{self.method}")
        setattr(self.target, self.method, self.org)
    
    def create_hook(self, fn: Callable):
        def hook(self_, *args, **kwargs):
            return fn(self_, self.org, *args, **kwargs)
        return hook

class Script(scripts.Script):

    def __init__(self):
        super().__init__()
    
    def title(self):
        return "Intm"
    
    def show(self, is_img2img):
        return True
    
    def ui(self, is_img2img):
        with gr.Group():
            show_input = gr.Checkbox(value=False, label="Show latent inputs.")
            show_output = gr.Checkbox(value=False, label="Show latent outputs.")
            show_each_step = gr.Checkbox(value=False, label="Show images which are generated in each step.")
        return [
            show_input,
            show_output,
            show_each_step,
        ]
    
    def run(self,
            p: StableDiffusionProcessing,
            input: bool,
            output: bool,
            step: bool,
    ):
        
        input_tensors: TensorResult = defaultdict(lambda: [])
        output_tensors: TensorResult = defaultdict(lambda: [])
        step_images: ImageResult = defaultdict(lambda: [])
        
        hooks = self.hook(
            p,
            p.sd_model, # type: ignore
            input_tensors,
            output_tensors,
            step_images
        )
        
        try:
            proc = process_images(p)
            
            b = putils.ProcessedBuilder()
            b.add_proc(proc)
            
            if input:
                for idx in sorted(input_tensors.keys()):
                    for step_idx, tensor in enumerate(input_tensors[idx]):
                        images = tensor_to_image(tensor, tensor.shape[0], 1)
                        for image in images:
                            b.add_ref(
                                idx,
                                image,
                                f"type=LATENT_INPUT, at={idx}, step={step_idx+1}"
                            )
                
            if output:
                for idx in output_tensors:
                    for step_idx, tensor in enumerate(output_tensors[idx]):
                        images = tensor_to_image(tensor, tensor.shape[0], 1)
                        for image in images:
                            b.add_ref(
                                idx,
                                image,
                                f"type=LATENT_OUTPUT, at={idx}, step={step_idx+1}"
                            )
            
            if step:
                for idx in output_tensors:
                    for step_idx, tensor in enumerate(output_tensors[idx]):
                        image_tensor = decode_first_stage(shared.sd_model, tensor.unsqueeze(0))
                        images = tensors_to_rgb_image(image_tensor)
                        for image in images:
                            b.add_ref(
                                idx,
                                image,
                                f"type=OUTPUT, at={idx}, step={step_idx+1}"
                            )
            
            #sample_to_image(decoded)
            #-> processing.decode_first_stage(shared.sd_model, sample)[0]
            return b.to_proc(p, proc)
        
        finally:
            for hook in hooks:
                try:
                    hook.remove()
                except:
                    pass
        
    def hook(self,
             p: StableDiffusionProcessing,
             sd_model: nn.Module,
             input: TensorResult,
             output: TensorResult,
             images: ImageResult
    ) -> list[Hooker]:
        
        #wrapper: nn.Module = getattr(sd_model, "model")
        ##unet = getattr(wrapper, "diffusion_model")
        ##
        ##def create_hook(input: bool, output: bool)
        #last_batch_no = -1
        #def diffusion_model_hooker(module: nn.Module, inputs: list[Tensor], outputs: Tensor):
        #    #import pdb; pdb.set_trace()
        #    #print("============================")
        #    #print(len(inputs))
        #    #print([x.shape for x in inputs])
        #    #print(outputs.shape)
        #    #print(state.sampling_step, state.sampling_steps, state.current_image_sampling_step)
        #    #print(state.job, state.job_no, state.job_count, state.job_timestamp)
        #    #print("============================")
        #    #input.append(inputs[0].detach().clone())
        #    #if step == p.steps:
        #    nonlocal last_batch_no
        #    batch_no = state.job_no
        #    step = state.sampling_step
        #    if batch_no == last_batch_no:
        #        step += 1
        #    last_batch_no = batch_no
        #    
        #    x, t = inputs
        #    assert x.shape[0] == outputs.shape[0], f"x.shape={x.shape}, outputs.shape={outputs.shape}"
        #    assert x.shape[1] == 4, f"x.shape={x.shape}"
        #    
        #    x, t, o = x[:p.batch_size], t[:p.batch_size], outputs[:p.batch_size]
        #    index0 = batch_no * p.batch_size
        #    for i, (inp, out) in enumerate(zip(x, o), index0):
        #        input[i].append(inp.detach().clone())
        #        #import pdb; pdb.set_trace()
        #        output[i].append(out.detach().clone())
        #
        #hook1 = wrapper.register_forward_hook(diffusion_model_hooker),
        
        def KDiffusionSampler_callback_state(self_, org, d, *args, **kwargs):
            result = org(self_, d, *args, **kwargs)
            # get C and UC combined latent
            batch_no = state.job_no
            xs: Tensor = d["x"]
            steps: int = d["i"] + 1
            latents: Tensor = d["denoised"]
            pos = batch_no * p.batch_size
            
            for i, (x, latent) in enumerate(zip(xs, latents), pos):
                input[i].append(x.detach().clone())
                output[i].append(latent.detach().clone())
            
            return result
        
        def VanillaStableDiffusionSampler_p_sample_ddim_hook(self_, org, x_dec, *args, **kwargs):
            batch_no = state.job_no
            pos = batch_no * p.batch_size
            
            res = org(self_, x_dec, *args, **kwargs)
            for i, (x, latent) in enumerate(zip(x_dec, res[1]), pos):
                input[i].append(x.detach().clone())
                output[i].append(latent.detach().clone())
            #print(self_.step, res.shape)
            return res
        
        hook2 = Hooker(KDiffusionSampler, "callback_state", KDiffusionSampler_callback_state)
        hook3 = Hooker(VanillaStableDiffusionSampler, "p_sample_ddim_hook", VanillaStableDiffusionSampler_p_sample_ddim_hook)
        #hooks = [hook1, hook2, hook3]
        hooks = [hook2, hook3]
        return hooks
        
def tensor_to_image(tensor: Tensor, grid_x: int, grid_y: int):
    assert len(tensor.shape) == 3
    
    max_ch, ih, iw = tensor.shape
    width = (grid_x * (iw + 1) - 1)
    height = (grid_y * (ih + 1) - 1)
    assert max_ch == grid_x * grid_y
    
    def each_slice(it: range, n: int):
        cur = []
        for x in it:
            cur.append(x)
            if n == len(cur):
                yield cur
                cur = []
        if 0 < len(cur):
            yield cur
    
    canvases: list[Image.Image] = []
    
    for chs in each_slice(range(max_ch), grid_x * grid_y):
        chs = list(chs)
        
        canvas = Image.new("L", (width, height), color=0)
        for iy in range(grid_y):
            if len(chs) == 0:
                break
            
            for ix in range(grid_x):
                if state.interrupted:
                    break
                
                if len(chs) == 0:
                    break
                
                ch = chs.pop(0)
                array = tensor[ch].cpu().numpy().astype(np.float32)
                
                # create image
                x = (iw+1) * ix
                y = (ih+1) * iy
                
                # sigmoid colorization
                array = 1.0 / (1.0 + np.exp(-array))
                array = np.clip(array * 256, 0, 255).astype(np.uint8)
                canvas.paste(Image.fromarray(array, "L"), (x, y))
        
        canvases.append(canvas)
    return canvases

def tensors_to_rgb_image(tensor: Tensor):
    assert len(tensor.shape) == 4
    
    return [
        tensor_to_rgb_image(tensor[i])
        for i in range(tensor.shape[0])
    ]

def tensor_to_rgb_image(tensor: Tensor):
    assert len(tensor.shape) == 3
    
    ch, h, w = tensor.shape
    assert ch == 3
    
    t = torch.clamp((tensor + 1.0) / 2.0, min=0.0, max=1.0)
    t = 255.0 * np.moveaxis(t.cpu().numpy(), 0, 2)
    return Image.fromarray(t.astype(np.uint8))
