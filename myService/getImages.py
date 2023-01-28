import os
import torch
import trimesh
import json
import numpy as np
from munch import *
from options import BaseOptions
from model import Generator
from generate_shapes_and_images import generate, generateImage
from render_video import render_video


class GetImages():
    def __init__(self) -> None:
        torch.random.manual_seed(234)
        # torch.random.manual_seed(321)
        self.device = "cuda"
        self.inference_identities = 10000

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

        # User options
        self.model_type = 'ffhq' # Whether to load the FFHQ or AFHQ model
        self.opt.inference.no_surface_renderings = True # When true, only RGB images will be created
        self.opt.inference.fixed_camera_angles = False # When true, each identity will be rendered from a specific set of 13 viewpoints. Otherwise, random views are generated
        self.opt.inference.identities = self.inference_identities # Number of identities to generate
        self.opt.inference.num_views_per_id = 1 # Number of viewpoints generated per identity. This option is ignored if self.opt.inference.fixed_camera_angles is true.

    def main(self):
        # main
        model_path = self.loadSavedModel()
        checkpoint = self.createResultDir(model_path)
        
        g_ema = self.loadImageGenerationModel(checkpoint)
        surface_g_ema = self.loadVolumnRender()
        
        mean_latent, surface_mean_latent = self.getMeanLatentVector(g_ema)
        prepareDatasetPath = "prepareDataset"

        with open(os.path.join(prepareDatasetPath , "json", "camera_paras.json"), 'w') as jsonFile:
            jsonFile.write("[")
        with open(os.path.join(prepareDatasetPath , "json", "sample_z.json"), 'w') as jsonFile:
            jsonFile.write("[")
        self.generate(g_ema, surface_g_ema, mean_latent, surface_mean_latent)
        with open(os.path.join(prepareDatasetPath , "json", "camera_paras.json"), 'a') as jsonFile:
            jsonFile.write("]")
        with open(os.path.join(prepareDatasetPath , "json", "sample_z.json"), 'a') as jsonFile:
            jsonFile.write("]")
        
        # store json
        # with open(os.path.join(prepareDatasetPath , "json", "camera_paras.json"), 'w') as jsonFile:
        #     json.dump(camera_paras_list , jsonFile)
        # with open(os.path.join(prepareDatasetPath , "json", "sample_z.json"), 'w') as jsonFile:
        #     json.dump(sample_z_list , jsonFile)

        if mean_latent:
            for i in range(len(mean_latent)):
                mean_latent[i] = mean_latent[i].tolist()
        with open(os.path.join(prepareDatasetPath , "json", "mean_latent.json"), 'w') as jsonFile:
            json.dump(mean_latent , jsonFile)
        
        # self.storeJson(camera_paras_list, os.path.join(prepareDatasetPath , "json", "camera_paras.json"))
        # self.storeJson(sample_z_list, os.path.join(prepareDatasetPath , "json", "sample_z"))
        # self.storeJson(mean_latent, "mean_latent")
        # self.storeJson(surface_g_ema, "surface_g_ema") # None
        # self.storeJson(surface_mean_latent, "surface_mean_latent") # None
        pass

    def loadSavedModel(self):
        # Load saved model
        if self.model_type == 'ffhq':
            model_path = 'ffhq1024x1024.pt'
            self.opt.model.size = 1024
            self.opt.experiment.expname = 'ffhq1024x1024'
        else:
            self.opt.inference.camera.azim = 0.15
            model_path = 'afhq512x512.pt'
            self.opt.model.size = 512
            self.opt.experiment.expname = 'afhq512x512'
        return model_path

    def createResultDir(self, model_path):
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
        checkpoint_path = os.path.join('full_models', model_path)
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    
    def loadImageGenerationModel(self, checkpoint):
        # Load image generation model
        g_ema = Generator(model_opt=self.opt.model, renderer_opt=self.opt.rendering, full_pipeline=True).to(self.device)
        self.pretrained_weights_dict = checkpoint["g_ema"]
        model_dict = g_ema.state_dict()
        for k, v in self.pretrained_weights_dict.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = v

        g_ema.load_state_dict(model_dict)

        return g_ema

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
            surface_g_ema = Generator(self.opt.surf_extraction.model, self.opt.surf_extraction.rendering, full_pipeline=False).to(self.device)

            # Load weights to surface extractor
            surface_extractor_dict = surface_g_ema.state_dict()
            for k, v in self.pretrained_weights_dict.items():
                if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                    surface_extractor_dict[k] = v

            surface_g_ema.load_state_dict(surface_extractor_dict)
        else:
            surface_g_ema = None
        
        return surface_g_ema
    
    def getMeanLatentVector(self, g_ema):
        # Get the mean latent vector for g_ema
        if self.opt.inference.truncation_ratio < 1:
            with torch.no_grad():
                mean_latent = g_ema.mean_latent(self.opt.inference.truncation_mean, self.device)
        else:
            surface_mean_latent = None

        # Get the mean latent vector for surface_g_ema
        if not self.opt.inference.no_surface_renderings:
            surface_mean_latent = mean_latent[0]
        else:
            surface_mean_latent = None

        return (mean_latent, surface_mean_latent)

    def generate(self, g_ema, surface_g_ema, mean_latent, surface_mean_latent):
        # locatoin = [0,0]
        generateImage(self.opt.inference, g_ema, surface_g_ema, self.device, mean_latent, surface_mean_latent)
        # camera_paras_list, sample_z_list = generate(self.opt.inference, g_ema, surface_g_ema, self.device, mean_latent, surface_mean_latent)
        # return (camera_paras_list, sample_z_list)
        pass
        
    def storeJson(self, obj, objName):
        newpath = './json/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        with open(f'{newpath}{objName}.json', 'w') as j:
            json.dump(obj , j)

if __name__ == "__main__":
    getImages = GetImages()
    getImages.main()