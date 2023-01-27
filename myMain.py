import warnings
warnings.filterwarnings("ignore")

import os
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
from generate_shapes_and_images import generate



from myService.myModel import *
from myService.myDataset import MyDataset
from myService.myUtils import my_collate
from myService.getImages import GetImages

# class #
class MyMain():
    def __init__(self):
        # config #
        self.use_cuda=True
        self.dataPath = './result/thumbs'
        self.model_save_dir = './result/model'
        self.learningRate = 0.01

        # set device #
        self.device = torch.device("cuda" if (torch.cuda.is_available() & self.use_cuda) else "cpu")
        print('===== MyMain Info =====')
        print(f'device: {self.device}')
        print()

        # Config Training List #
        # Diff Encoders
        self.initModel()
        self.initLoss()
        self.initOptmizer()
        self.encoderUsed_list_map = {
            "simpleMLP": self.model_mlp,
            "CNN": self.model_cnn,
        }

    def saveModel(self, model, modelName):
        model_path = os.path.join(self.model_save_dir, f'{modelName}.ckpt')
        torch.save(model.state_dict(), model_path)
        print(f'Saved {modelName} into {self.model_save_dir}...')

    def loadModelState(self, model, modelName):
        print(f'Loading the trained models {modelName}')
        model_path = os.path.join(self.model_save_dir, f'{modelName}.ckpt')
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    def initModel(self):
        self.model_mlp = MLP(n_class=3).to(self.device)
        self.model_convNet = ConvNet(n_class=3).to(self.device)
        self.model_cnn = CNN().to(self.device)

    def initLoss(self):
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)

    def initOptmizer(self):
        self.optimizer = optim.Adam(self.model_cnn.parameters(), lr=self.learningRate)

    def getDataLoader(self, batch_size=5, transform =None):
        # transform = transforms.Compose([
        #         transforms.Resize((28,28)),
        #         transforms.ToTensor()
        #         ])
        
        mydataset = MyDataset(transform = transform)
        mydata_loader = DataLoader(mydataset, batch_size=batch_size, num_workers=0,  collate_fn = my_collate)
        
        checkShape = False
        if checkShape:
            for data in mydata_loader:
                print(len(data))
                print(data[0].shape)
                print(data[1].shape)
                break 
        return mydata_loader

    def plot(self, plt_loss_train):
        plt.plot(list(range(len(plt_loss_train))), plt_loss_train)
        plt.savefig('./result/loss.png')

    def DiffTrains(self):
        """
        ===== Training Info =====
        Current Pipeline: image_64x64 -(InverseNet)-> z_1x265 (see loss)

        ===== Dev Info =====
        handle generate image (used comment for above dev)

        """
        # Config Training List #
        # Diff Encoders
        encoderUsed_list = ["CNN"]

        # Diff Ways of Loss Function Calculation
        lossWeight_list = [
            {
                "latent": 1,
                "camera_para": 0,
                "image": 0
            }
        ]
        # lossProcess_list = [["latent", "camera_para", "image"]]
        lossProcess_list = [
            [["latent"]],
            # [["latent", "camera_para", "image"]]
        ]
        lossSwitchMoment_list = [("epoch", 10)]
        
        # Diff HyperParameter
        batchSize_list = [5]
        epoch_list = [100]

        total_info = {
            "encoderUsed_list": encoderUsed_list,
            "lossWeight_list": lossWeight_list,
            "lossProcess_list": lossProcess_list,
            "lossSwitchMoment_list": lossSwitchMoment_list,
            "batchSize_list": batchSize_list,
            "epoch_list": epoch_list,
        }

        # Main Loop #
        plt_loss_train_info_list = []

        # Diff Encoders #
        for encoderUsed in encoderUsed_list:

            # Diff Ways of Loss Function Calculation #
            for lossWeight in lossWeight_list:
                for lossProcess in lossProcess_list:
                    for lossSwitchMoment in lossSwitchMoment_list:
                        
                        # Diff HyperParameter #
                        for batchSize in batchSize_list:
                            for totalEpoch in epoch_list:
                                
                                training_paras = {
                                    "encoderUsed": encoderUsed,
                                    "lossWeight": lossWeight,
                                    "lossProcess": lossProcess,
                                    "lossSwitchMoment": lossSwitchMoment,
                                    "batchSize": batchSize,
                                    "totalEpoch": totalEpoch
                                }
                                plt_loss_train = self.train(**training_paras)

                                for i in range(len(plt_loss_train)):
                                    plt_loss_train[i] = plt_loss_train[i].cpu().tolist()

                                plt_loss_train_info = {
                                    **training_paras,
                                    "plt_loss_train": plt_loss_train,
                                }
                                plt_loss_train_info_list.append(plt_loss_train_info)
                            # for epoch_list end

        with open('result/plt_loss_train_info_list.json', 'w') as f:
            json.dump(plt_loss_train_info_list, f)
        with open('result/total_info.json', 'w') as f:
            json.dump(total_info, f)

        return (plt_loss_train_info_list, total_info)
    
    def train(
        self,
        encoderUsed,
        lossWeight,
        lossProcess,
        lossSwitchMoment,
        batchSize,
        totalEpoch
    ):
        """
        config

        """
        encoder = self.encoderUsed_list_map[encoderUsed]
        transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.ToTensor()
                ])
        dataLoader = self.getDataLoader(batch_size=batchSize, transform=transform)

        print('===== Training Info =====')
        plt_loss_train=[]
        for epochIdx in range(totalEpoch):

            CurrentEpochLossFunctionIdx = int((epochIdx / lossSwitchMoment[1]) % len(lossProcess))
            CurrentEpochLossFunction_list = lossProcess[CurrentEpochLossFunctionIdx]
            
            encoder.train()
            train_loss = 0
            
            for batch_idx, (data, target) in enumerate(dataLoader):

                data = torch.squeeze(data)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                
                """
                get predict result

                """                
                latent_predict = encoder(data)
                # generated_image = generate(self.opt.inference, g_ema, surface_g_ema, self.device, mean_latent, surface_mean_latent)
                
                """
                calculate loss

                """
                latent_loss = self.loss(latent_predict,target)  
                
                if "latent" in CurrentEpochLossFunction_list: 
                    train_loss += latent_loss*lossWeight["latent"]
                # if "camera_para" in CurrentEpochLossFunction_list: 
                #     train_loss += camera_para_loss*lossWeight["camera_para"]
                # if "image" in CurrentEpochLossFunction_list: 
                #     train_loss += image_loss*lossWeight["image"]

                # surface_mean_latent = None
                # mean_latent = None

                """
                update parameters
                
                """
                latent_loss.backward()
                self.optimizer.step()
            # for dataLoader end

            train_loss /= len(dataLoader.dataset)
            plt_loss_train.append(train_loss)
        
            if epochIdx % 10 ==0:
                print(f'{encoderUsed}[epochIdx: {epochIdx+1}/{totalEpoch}], Average loss (Train):{train_loss}')
        # for totalEpoch end 

        print(f'{encoderUsed}[epochIdx: {epochIdx+1}/{totalEpoch}], Average loss (Train):{train_loss}')
        print('training done.')
        print()

        return plt_loss_train

    def main(self):
        # self.initModel()
        # self.initLoss()
        # self.initOptmizer()
        # plt_loss_train_info_list, total_info = self.DiffTrains()
        # self.plot(plt_loss_train_info_list)
        
        plt_loss_train_info_list, total_info = self.DiffTrains()

if __name__ == "__main__":
    generateImage = True
    train = False

    if generateImage:
        getImages = GetImages()
        getImages.main()

    if train:
        myMain = MyMain()
        myMain.main()
   
