from PIL import Image, ImageEnhance
import numpy as np
import cv2
import torch
import os
from skimage.util import random_noise
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from bm3d import bm3d_rgb
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor

class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError
        
class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents,
                                   head_start_step=50 - max(self.noise_step // 20, 1),
                                   guidance_scale=7.5,
                                   generator=generator, )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    img.save(out)

            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                img = Image.open(img_path)
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=torch.float16, device=self.device).permute(2, 0, 1).unsqueeze(0)
                latents = self.pipe.vae.encode(img).latent_dist
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
                noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(torch.half)
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                if return_latents:
                    ret_latents.append(latents.cpu())

            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            if return_latents:
                return ret_latents