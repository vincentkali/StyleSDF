import json
import torch
import PIL.Image as Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class MyDataset(Dataset):
    '''
    load the dataset
    '''
    def __init__(self, transform = None):
        camera_json_path = '.\json\camera_paras.json'
        z_json_path = '.\json\sample_z.json'
        
        default_transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor()
            ])

        with open(camera_json_path) as camera_json:
            camera_paras = json.load(camera_json)
        with open(z_json_path) as z_json:
            z = json.load(z_json)
        
        self.camera_paras = camera_paras
        self.z = z

        if transforms == None:
            self.transform = default_transform
        else:
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
        folder_path = './result/thumbs'
        image_name = folder_path + "/" + str(idx).rjust(7, "0") + "_thumb.png"
        image = Image.open(image_name, mode='r')
        image = image.convert('RGB')

        # camera_paras = self.camera_paras[idx]["sample_cam_extrinsics"]
        z = self.z[idx][0]
        
        # Convert PIL label image to torch.Tensor
        image = self.transform(image)
        
        # label = [camera_paras, z]
        # label = torch.tensor(label)
        # camera_paras = torch.tensor(camera_paras)
        z = torch.tensor(z)

        return (image, z)