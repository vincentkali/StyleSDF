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
torch.random.manual_seed(321)

"""
config

"""
device = "cuda"
inference_identities = 10


opt = BaseOptions().parse()
opt.camera.uniform = True
opt.model.is_test = True
opt.model.freeze_renderer = False
opt.rendering.offset_sampling = True
opt.rendering.static_viewdirs = True
opt.rendering.force_background = True
opt.rendering.perturb = 0
opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
opt.inference.style_dim = opt.model.style_dim
opt.inference.project_noise = opt.model.project_noise

print("Success import lib and set opt")

# User options
model_type = 'ffhq' # Whether to load the FFHQ or AFHQ model
opt.inference.no_surface_renderings = True # When true, only RGB images will be created
opt.inference.fixed_camera_angles = False # When true, each identity will be rendered from a specific set of 13 viewpoints. Otherwise, random views are generated
opt.inference.identities = inference_identities # Number of identities to generate
opt.inference.num_views_per_id = 1 # Number of viewpoints generated per identity. This option is ignored if opt.inference.fixed_camera_angles is true.

# Load saved model
if model_type == 'ffhq':
    model_path = 'ffhq1024x1024.pt'
    opt.model.size = 1024
    opt.experiment.expname = 'ffhq1024x1024'
else:
    opt.inference.camera.azim = 0.15
    model_path = 'afhq512x512.pt'
    opt.model.size = 512
    opt.experiment.expname = 'afhq512x512'

# Create results directory
result_model_dir = 'final_model'
results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir)
if opt.inference.fixed_camera_angles:
    opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'fixed_angles')
else:
    opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'random_angles')

os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
os.makedirs(os.path.join(opt.inference.results_dst_dir, 'images'), exist_ok=True)
if not opt.inference.no_surface_renderings:
    os.makedirs(os.path.join(opt.inference.results_dst_dir, 'depth_map_meshes'), exist_ok=True)
    os.makedirs(os.path.join(opt.inference.results_dst_dir, 'marching_cubes_meshes'), exist_ok=True)

opt.inference.camera = opt.camera
opt.inference.size = opt.model.size
checkpoint_path = os.path.join('full_models', model_path)
checkpoint = torch.load(checkpoint_path)

# Load image generation model
g_ema = Generator(opt.model, opt.rendering).to(device)
pretrained_weights_dict = checkpoint["g_ema"]
model_dict = g_ema.state_dict()
for k, v in pretrained_weights_dict.items():
    if v.size() == model_dict[k].size():
        model_dict[k] = v

g_ema.load_state_dict(model_dict)

# Load a second volume renderer that extracts surfaces at 128x128x128 (or higher) for better surface resolution
if not opt.inference.no_surface_renderings:
    opt['surf_extraction'] = Munch()
    opt.surf_extraction.rendering = opt.rendering
    opt.surf_extraction.model = opt.model.copy()
    opt.surf_extraction.model.renderer_spatial_output_dim = 128
    opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
    opt.surf_extraction.rendering.return_xyz = True
    opt.surf_extraction.rendering.return_sdf = True
    surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(device)


    # Load weights to surface extractor
    surface_extractor_dict = surface_g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
            surface_extractor_dict[k] = v

    surface_g_ema.load_state_dict(surface_extractor_dict)
else:
    surface_g_ema = None

# Get the mean latent vector for g_ema
if opt.inference.truncation_ratio < 1:
    with torch.no_grad():
        mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
else:
    surface_mean_latent = None

# Get the mean latent vector for surface_g_ema
if not opt.inference.no_surface_renderings:
    surface_mean_latent = mean_latent[0]
else:
    surface_mean_latent = None

print("Success load model")

print(f'opt.inference: {opt.inference}')
print(f'g_ema: {g_ema}')
print(f'surface_g_ema: {surface_g_ema}')
print(f'device: {device}')
print(f'mean_latent: {mean_latent}')
print(f'surface_mean_latent: {surface_mean_latent}')

camera_paras_list, sample_z_list = generate(opt.inference, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent)

newpath = './json'

if not os.path.exists(newpath):
    os.makedirs(newpath)
with open('json/camera_paras.json', 'w') as camera_paras:
    json.dump(camera_paras_list , camera_paras)

with open('json/sample_z.json', 'w') as sample_z:
    json.dump(sample_z_list , sample_z)

print("Success generate images")