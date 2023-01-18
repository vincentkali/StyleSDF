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
        print('===== MyMain Info =====')
        print(f'device: {self.device}')
        print()

    def main(self):
        mydata_loader = self.getDataLoader()
        self.initModel()
        self.initLoss()
        self.initOptmizer()
        plt_loss_train = self.train(mydata_loader)
        self.plot(plt_loss_train)

    def getDataLoader(self):
        mytransform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.ToTensor()
                ])
        
        mydataset = MyDataset(transforms_ = mytransform)
        mydata_loader = DataLoader(mydataset, batch_size=5, num_workers=0,  collate_fn = my_collate)
        
        checkShape = False
        if checkShape:
            for data in mydata_loader:
                print(len(data))
                print(data[0].shape)
                print(data[1].shape)
                break 
        return mydata_loader
    
    def initModel(self):
        self.model_mlp = MLP(n_class=3).to(self.device)
        self.model_cnn = ConvNet(n_class=3).to(self.device)

    def initLoss(self):
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)

    def initOptmizer(self):
        self.optimizer_mlp = optim.Adam(self.model_mlp.parameters(), lr=0.01)

    def train(self, mydata_loader):
        """
        image_64x64 -(InverseNet)-> z_1x265 (see loss)
        """
        print('===== Training Info =====')
        total_epoch=100
        plt_loss_train=[]
        
        for epoch in range(total_epoch):
            # train
            self.model_mlp.train()
            train_loss_mlp = 0
            
            for batch_idx, (data, target) in enumerate(mydata_loader):
                data = torch.squeeze(data)
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer_mlp.zero_grad()
                output_mlp = self.model_mlp(data)
                
                loss_mlp = self.loss(output_mlp,target)  
                train_loss_mlp += loss_mlp

                loss_mlp.backward()
                self.optimizer_mlp.step()

            train_loss_mlp /= len(mydata_loader.dataset)
            plt_loss_train.append(train_loss_mlp)
        
            if epoch % 10 ==0:
                print('MLP[epoch: {}/{}], Average loss (Train):{:.10f}'.format(
                    epoch+1, total_epoch, train_loss_mlp))
        
        print('MLP[epoch: {}/{}], Average loss (Train):{:.10f}'.format(
                    epoch+1, total_epoch, train_loss_mlp))
        print('training done.')
        print()

        for i in range(len(plt_loss_train)):
            plt_loss_train[i] = plt_loss_train[i].cpu().tolist()
        return plt_loss_train

    def plot(self, plt_loss_train):
        plt.plot(list(range(len(plt_loss_train))), plt_loss_train)
        plt.savefig('./result/loss.png')
        # plt.show()  

if __name__ == "__main__":
    # getImages = GetImages()
    # getImages.main()

    myMain = MyMain()
    myMain.main()
   
