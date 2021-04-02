# %%
import torch
from torchsummary import summary
from model.pytorch_ssim import SSIM
import numpy as np

class SI_DAE(torch.nn.Module):
    def __init__(self):
        super(SI_DAE, self).__init__()
        self.Encoder_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1),  
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2))
        self.Encoder_maxpool1 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Encoder_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 50, 3, 1, 1),  
            torch.nn.BatchNorm2d(50),
            torch.nn.LeakyReLU(0.2))
        self.Encoder_maxpool2 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Encoder_conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(50, 36, 3, 1, 1),
            torch.nn.BatchNorm2d(36),
            torch.nn.LeakyReLU(0.2))
        self.Encoder_maxpool3 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Encoder_conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(36,22, 3, 1, 1),  
            torch.nn.BatchNorm2d(22),
            torch.nn.LeakyReLU(0.2))
        self.Encoder_maxpool4 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Encoder_conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(22, 8, 3, 1, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Sigmoid())
        self.Encoder_maxpool5 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.Decoder_convtrans1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 22, 3, 1, 1),)
        self.Decoder_maxunpool1=torch.nn.MaxUnpool2d(2,stride=2)
        self.Decoder_convtrans2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(22, 36, 3, 1, 1),)
        self.Decoder_maxunpool2=torch.nn.MaxUnpool2d(2,stride=2)
        self.Decoder_convtrans3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(36, 50, 3, 1, 1),)
        self.Decoder_maxunpool3=torch.nn.MaxUnpool2d(2,stride=2)
        self.Decoder_convtrans4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(50, 64, 3, 1, 1),)
        self.Decoder_maxunpool4=torch.nn.MaxUnpool2d(2,stride=2)
        self.Decoder_convtrans5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 1, 3, 1, 1),)
        self.Decoder_maxunpool5=torch.nn.MaxUnpool2d(2,stride=2)
         
    def forward(self, inputs):
        if self.training:
            std=0.01
        else:
            std=0.0
        noise=torch.tensor(np.random.normal(loc=0, scale=std, size=inputs.shape))
        noise_inputs=inputs+noise.type(torch.FloatTensor).to(inputs.device)
        x = self.Encoder_conv1(noise_inputs)
        x , indices1 = self.Encoder_maxpool1(x)
        x = self.Encoder_conv2(x)
        x , indices2 = self.Encoder_maxpool2(x)
        x = self.Encoder_conv3(x)
        x , indices3 = self.Encoder_maxpool3(x)
        x = self.Encoder_conv4(x)
        x , indices4 = self.Encoder_maxpool4(x)
        x = self.Encoder_conv5(x)
        feature , indices5 = self.Encoder_maxpool5(x)
        
        x = self.Decoder_maxunpool1(feature,indices5)
        x = self.Decoder_convtrans1(x)
        x = self.Decoder_maxunpool2(x,indices4)
        x = self.Decoder_convtrans2(x)
        x = self.Decoder_maxunpool3(x,indices3)
        x = self.Decoder_convtrans3(x)
        x = self.Decoder_maxunpool4(x,indices2)
        x = self.Decoder_convtrans4(x)
        x = self.Decoder_maxunpool5(x,indices1)
        x = self.Decoder_convtrans5(x)
        return x,feature,torch.nn.MSELoss()(x,inputs),1-SSIM()(x,inputs)
    
class DI_DAE(torch.nn.Module):
    def __init__(self):
        super(DI_DAE, self).__init__()
        self.Encoder_conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1),  
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2))
        self.Encoder_maxpool1 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Encoder_conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 50, 3, 1, 1),  
            torch.nn.BatchNorm2d(50),
            torch.nn.LeakyReLU(0.2))
        self.Encoder_maxpool2 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Encoder_conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(50, 36, 3, 1, 1),
            torch.nn.BatchNorm2d(36),
            torch.nn.LeakyReLU(0.2))
        self.Encoder_maxpool3 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Encoder_conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(36,22, 3, 1, 1),  
            torch.nn.BatchNorm2d(22),
            torch.nn.LeakyReLU(0.2))
        self.Encoder_maxpool4 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        self.Encoder_conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(22, 8, 3, 1, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Sigmoid())
        self.Encoder_maxpool5 = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
        
        self.Decoder_convtrans1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 22, 3, 1, 1),)
        self.Decoder_maxunpool1=torch.nn.MaxUnpool2d(2,stride=2)
        self.Decoder_convtrans2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(22, 36, 3, 1, 1),)
        self.Decoder_maxunpool2=torch.nn.MaxUnpool2d(2,stride=2)
        self.Decoder_convtrans3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(36, 50, 3, 1, 1),)
        self.Decoder_maxunpool3=torch.nn.MaxUnpool2d(2,stride=2)
        self.Decoder_convtrans4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(50, 64, 3, 1, 1),)
        self.Decoder_maxunpool4=torch.nn.MaxUnpool2d(2,stride=2)
        self.Decoder_convtrans5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 1, 3, 1, 1),)
        self.Decoder_maxunpool5=torch.nn.MaxUnpool2d(2,stride=2)
        
    def forward(self, inputs):
        if self.training:
            std=0.01
        else:
            std=0.0
        noise=torch.tensor(np.random.normal(loc=0, scale=std, size=inputs.shape))
        noise_inputs=inputs+noise.type(torch.FloatTensor).to(inputs.device)
        x = self.Encoder_conv1(noise_inputs)
        x , indices1 = self.Encoder_maxpool1(x)
        x = self.Encoder_conv2(x)
        x , indices2 = self.Encoder_maxpool2(x)
        x = self.Encoder_conv3(x)
        x , indices3 = self.Encoder_maxpool3(x)
        x = self.Encoder_conv4(x)
        x , indices4 = self.Encoder_maxpool4(x)
        x = self.Encoder_conv5(x)
        feature , indices5 = self.Encoder_maxpool5(x)
        
        x = self.Decoder_maxunpool1(feature,indices5)
        x = self.Decoder_convtrans1(x)
        x = self.Decoder_maxunpool2(x,indices4)
        x = self.Decoder_convtrans2(x)
        x = self.Decoder_maxunpool3(x,indices3)
        x = self.Decoder_convtrans3(x)
        x = self.Decoder_maxunpool4(x,indices2)
        x = self.Decoder_convtrans4(x)
        x = self.Decoder_maxunpool5(x,indices1)
        x = self.Decoder_convtrans5(x)
        return x,feature,torch.nn.MSELoss()(x,inputs),1-SSIM()(x,inputs)
# %%
if __name__ == '__main__':
    device="cuda"
    AE = SI_DAE().to(device).eval()
    img=torch.ones((10,1,64,64)).to(device)
    x,feature,mse,ssim=AE(img)
    summary(AE,input_size=(1,64,64))
    
# %%
