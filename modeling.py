from turtle import forward
import torch.nn as nn
from torchvision.models import resnet18



class encoders(nn.Module):
    def __init__(self):
        super(encoders, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=3, padding=1),  # b, 3, 10, 10
            nn.ReLU(True),
        # b, 16, 5, 5
            nn.Conv2d(32, 64, 3, stride=3, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),

            nn.Conv2d(64, 32, 3, stride=3, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),# b, 8, 2, 2
            # b, 8, 2, 2
            
        )
    def forward(self, x):
        x = self.encoder(x)
        return x

class decoders(nn.Module):
    def __init__(self):
        super(decoders, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 5, stride=3, padding=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 5, stride=3, padding=0),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=3, padding=4),  # b, 1, 28, 28
            nn.Tanh())
    def forward(self, x):
        x = self.decoder(x)
        return x        




class OUR_model(nn.Module):
    def __init__(self):
        super(OUR_model,self).__init__()

        self.classifier = resnet18()
        self.encoder = encoders()
        self.decoder = decoders()
        self.Linear = nn.Linear(1000,3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        conf_level = self.Linear(self.classifier(x))
        # conf_level = self.softmax(conf_level) 
        # malignant = torch.argmax(conf_level,dim=1) 
        #malignant = malignant.view(-1,1)# batch*1 #loss 1
        #malignant = torch.unsqueeze(malignant,2)
        #malignant = torch.unsqueeze(malignant,3) # batch*1*1*1

        # ad_x = malignant*x #batch*1*1*1 X batch*3*1024*1024

        output_encoder = self.encoder(x) #latent factor
        output_decoder = self.decoder(output_encoder) # loss 2



        return conf_level,output_encoder,output_decoder 