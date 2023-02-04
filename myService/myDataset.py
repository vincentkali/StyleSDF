import json
import torch
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


class MyDataset(Dataset):
    '''
    load the dataset
    '''
    def __init__(self, transform = None):
        camera_json_path = './prepareDataset/json/camera_paras.json'
        z_json_path = './prepareDataset/json/sample_z_actual_used.json'
        
        default_transform = transforms.Compose([
            # transforms.Resize((28,28)),
            transforms.ToTensor()
            ])

        with open(camera_json_path) as jsonFile:
            camera_paras = json.load(jsonFile)

        with open(z_json_path) as jsonFile:
            sample_z = json.load(jsonFile)
        
        self.camera_paras = camera_paras
        self.sample_z = sample_z

        if transform == None:
            print("default")
            self.transform = default_transform
        else:
            print("not default")
            self.transform = transform
        print('===== MyDataset Info =====')
        print('number of total data:{}'.format(len(self.camera_paras)))
        print()

    def __len__(self):
        return len(self.camera_paras)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        # read image with PIL module
        image_name = './prepareDataset/thumbs/' + str(idx).rjust(7, "0") + ".png"
        image = Image.open(image_name, mode='r')
        image = image.convert('RGB')
        image = self.transform(image)

        # len: 4
        sample_z = self.sample_z[idx][0] 
        
        # len: 14
        camera_paras = np.array(self.camera_paras[idx]["sample_cam_extrinsics"][0]).flatten().tolist() + self.camera_paras[idx]["sample_locations"][0]

        # len: 18
        target = sample_z + camera_paras
        target = torch.tensor(target)

        return (image, target)