import warnings
warnings.filterwarnings("ignore")

import json
import torch
from torch import optim
import numpy as np
from torchvision import datasets, transforms
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import subprocess


from myService.myModel import ConvM, ConvNet, MLP
from myService.myDataset import MyDataset
from myService.myUtils import my_collate
from myService.getImages import GetImages

# class #
class MyMain():
    def __init__(self):
        # config #
        self.use_cuda=True
        self.dataPath = '.\result\thumbs'

        # set device #
        self.device = torch.device("cuda" if (torch.cuda.is_available() & self.use_cuda) else "cpu")
        print(f'device: {self.device}')

    def main(self):
        self.setDataLoader()
        self.other()
        self.train()

    def setDataLoader(self):
        # set data loader #
        mytransform = transforms.Compose([
                # transforms.Resize((28,28)),
                transforms.ToTensor()
                ])
        mydataset = MyDataset(transforms_ = mytransform)
        self.mydata_loader = DataLoader(mydataset, batch_size=5, num_workers=0,  collate_fn = my_collate)
        for data in self.mydata_loader:
            print(len(data))
            print(data[0].shape)
            print(data[1].shape)
            break 
    
    def other(self):
        self.model_mlp = MLP(n_class=3)
        print(self.model_mlp)
        self.model_mlp = self.model_mlp.to(self.device)

            
        # initialize the MLP
        self.model_mlp = MLP(n_class=3).to(self.device)

        # 步驟3. loss function宣告
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)

        # 步驟4. optimator宣告
        self.optimizer_mlp = optim.Adam(self.model_mlp.parameters(), lr=0.01)

    def train(self):
        # 步驟5. 模型開始訓練
        total_epoch=100
        plt_loss_train=[]
        for epoch in range(total_epoch):
            # train
            self.model_mlp.train()
            train_loss_mlp = 0
            for batch_idx, (data, target) in enumerate(self.mydata_loader):
                data = torch.squeeze(data)
                data, target = data.to(self.device), target.to(self.device)
                # MLP
        #         print(target)
                self.optimizer_mlp.zero_grad()
                output_mlp = self.model_mlp(data)
                loss_mlp = self.loss(output_mlp,target)  
                train_loss_mlp += loss_mlp
                loss_mlp.backward()
                self.optimizer_mlp.step()
            train_loss_mlp /= len(self.mydata_loader.dataset)
            plt_loss_train.append(train_loss_mlp)
        
            if epoch % 10 ==0:
                print('MLP[epoch: {}/{}], Average loss (Train):{:.10f}'.format(
                    epoch+1, total_epoch, train_loss_mlp))
        print('MLP[epoch: {}/{}], Average loss (Train):{:.10f}'.format(
                    epoch+1, total_epoch, train_loss_mlp))
        print('training done.')

if __name__ == "__main__":
    getImages = GetImages()
    getImages.main()

    # myMain = MyMain()
    # myMain.main()
   
