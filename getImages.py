import os
import torch
import trimesh
import json
import numpy as np
from munch import *
from options import BaseOptions
from model import Generator
from generate_shapes_and_images import generate
from render_video import render_video

class GetImages():
    def __init__(self) -> None:
        torch.random.manual_seed(234)
        # torch.random.manual_seed(321)
        self.device = "cuda"
        self.inference_identities = 10

        self.opt = BaseOptions().parse()
        self.opt.camera.uniform = True
        self.opt.model.is_test = True
        self.opt.model.freeze_renderer = False
        self.opt.rendering.offset_sampling = True
        self.opt.rendering.static_viewdirs = True
        self.opt.rendering.force_background = True
        self.opt.rendering.perturb = 0
        self.opt.inference.renderer_output_size = self.opt.model.renderer_spatial_output_dim
        self.opt.inference.style_dim = self.opt.model.style_dim
        self.opt.inference.project_noise = self.opt.model.project_noise
        print("Success import lib and set self.opt")

        # User options
        self.model_type = 'ffhq' # Whether to load the FFHQ or AFHQ model
        self.opt.inference.no_surface_renderings = True # When true, only RGB images will be created
        self.opt.inference.fixed_camera_angles = False # When true, each identity will be rendered from a specific set of 13 viewpoints. Otherwise, random views are generated
        self.opt.inference.identities = self.inference_identities # Number of identities to generate
        self.opt.inference.num_views_per_id = 1 # Number of viewpoints generated per identity. This option is ignored if self.opt.inference.fixed_camera_angles is true.

    def main(self):
        self.loadSavedModel()
        self.createResultDir()
        self.loadImageGenerationModel()
        self.loadVolumnRender()
        self.getMeanLatentVector()
        self.generate()
        self.storeJson()

    def loadSavedModel(self):
        # Load saved model
        if self.model_type == 'ffhq':
            self.model_path = 'ffhq1024x1024.pt'
            self.opt.model.size = 1024
            self.opt.experiment.expname = 'ffhq1024x1024'
        else:
            self.opt.inference.camera.azim = 0.15
            self.model_path = 'afhq512x512.pt'
            self.opt.model.size = 512
            self.opt.experiment.expname = 'afhq512x512'

    def createResultDir(self):
        # Create results directory
        result_model_dir = 'final_model'
        results_dir_basename = os.path.join(self.opt.inference.results_dir, self.opt.experiment.expname)
        self.opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir)
        if self.opt.inference.fixed_camera_angles:
            self.opt.inference.results_dst_dir = os.path.join(self.opt.inference.results_dst_dir, 'fixed_angles')
        else:
            self.opt.inference.results_dst_dir = os.path.join(self.opt.inference.results_dst_dir, 'random_angles')

        os.makedirs(self.opt.inference.results_dst_dir, exist_ok=True)
        os.makedirs(os.path.join(self.opt.inference.results_dst_dir, 'images'), exist_ok=True)
        if not self.opt.inference.no_surface_renderings:
            os.makedirs(os.path.join(self.opt.inference.results_dst_dir, 'depth_map_meshes'), exist_ok=True)
            os.makedirs(os.path.join(self.opt.inference.results_dst_dir, 'marching_cubes_meshes'), exist_ok=True)

        self.opt.inference.camera = self.opt.camera
        self.opt.inference.size = self.opt.model.size
        checkpoint_path = os.path.join('full_models', self.model_path)
        self.checkpoint = torch.load(checkpoint_path)
    
    def loadImageGenerationModel(self):
        # Load image generation model
        self.g_ema = Generator(self.opt.model, self.opt.rendering).to(self.device)
        self.pretrained_weights_dict = self.checkpoint["g_ema"]
        model_dict = self.g_ema.state_dict()
        for k, v in self.pretrained_weights_dict.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        self.g_ema.load_state_dict(model_dict)

    def loadVolumnRender(self):
        # Load a second volume renderer that extracts surfaces at 128x128x128 (or higher) for better surface resolution
        if not self.opt.inference.no_surface_renderings:
            self.opt['surf_extraction'] = Munch()
            self.opt.surf_extraction.rendering = self.opt.rendering
            self.opt.surf_extraction.model = self.opt.model.copy()
            self.opt.surf_extraction.model.renderer_spatial_output_dim = 128
            self.opt.surf_extraction.rendering.N_samples = self.opt.surf_extraction.model.renderer_spatial_output_dim
            self.opt.surf_extraction.rendering.return_xyz = True
            self.opt.surf_extraction.rendering.return_sdf = True
            self.surface_g_ema = Generator(self.opt.surf_extraction.model, self.opt.surf_extraction.rendering, full_pipeline=False).to(self.device)

            # Load weights to surface extractor
            surface_extractor_dict = self.surface_g_ema.state_dict()
            for k, v in self.pretrained_weights_dict.items():
                if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                    surface_extractor_dict[k] = v

            self.surface_g_ema.load_state_dict(surface_extractor_dict)
        else:
            self.surface_g_ema = None
    
    def getMeanLatentVector(self):
        # Get the mean latent vector for self.g_ema
        if self.opt.inference.truncation_ratio < 1:
            with torch.no_grad():
                self.mean_latent = self.g_ema.mean_latent(self.opt.inference.truncation_mean, self.device)
        else:
            self.surface_mean_latent = None

        # Get the mean latent vector for surface_g_ema
        if not self.opt.inference.no_surface_renderings:
            self.surface_mean_latent = self.mean_latent[0]
        else:
            self.surface_mean_latent = None

        print("Success load model")

        print(f'opt.inference: {self.opt.inference}')
        print(f'g_ema: {self.g_ema}')
        print(f'surface_g_ema: {self.surface_g_ema}')
        print(f'device: {self.device}')
        print(f'mean_latent: {self.mean_latent}')
        print(f'surface_mean_latent: {self.surface_mean_latent}')

    def generate(self):
        self.camera_paras_list, self.sample_z_list = generate(self.opt.inference, self.g_ema, self.surface_g_ema, self.device, self.mean_latent, self.surface_mean_latent)

    def storeJson(self):
        newpath = './json'

        if not os.path.exists(newpath):
            os.makedirs(newpath)
        with open('json/camera_paras.json', 'w') as camera_paras:
            json.dump(self.camera_paras_list , camera_paras)

        with open('json/sample_z.json', 'w') as sample_z:
            json.dump(self.sample_z_list , sample_z)

if __name__ == "__main__":
    getImages = GetImages()
    getImages.main()