from torchvision.io import read_image
import torch.nn as nn
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import show_image
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from dataset import NMDataset
from training import Training,validate
from modeling import encoders,decoders,OUR_model


transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([1024,1024]),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])


# create dataset 
print("__________Create Dataset___________")
train_set = NMDataset('train',transformations)
valid_set = NMDataset('val',transformations)
test_set = NMDataset('test',transformations)


train_loader = DataLoader(train_set, batch_size = 2, shuffle = True)
valid_loader = DataLoader(valid_set, batch_size = 2, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 2, shuffle = True)




# create view_data
for data, label in train_loader:
  view_data = data[0:1]
  break


# model prepare
model = OUR_model().cuda()
Loss1 = nn.CrossEntropyLoss()
Loss2 = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


print("____________Training_____________")
num_epochs = 10
import tqdm
for epoch in range(num_epochs):
    start_time = time.perf_counter()
    # loss1=0

    # loss2=0
    loss,loss1,loss2= Training(model,Loss1,Loss2,optimizer, test_loader)
    val_loss = validate(model,Loss1,Loss2, valid_loader)
    end_time = time.perf_counter()
    print('Epoch : {}, loss:{:.4f}, loss1:{:.4f},loss2:{:.4f},val_loss:{:.4f}, {:.2f} seconds.'.format(epoch+1, loss, loss1,loss2,val_loss,end_time-start_time))

    if epoch % 5 == 0:
        output_decoders = []
        conf_level,output_encoder,output_decoder = model(view_data.cuda())  
        output_decoders.append(output_decoder)
        # f, axarr = plt.subplots(2,2,figsize= (10,8))
        # n = 0
        # for out_data, test_data in zip(output_decoder,view_data):
        #     out_data = out_data.permute(1,2,0).cpu().detach().numpy()   
        #     test_data  = test_data.permute(1,2,0).cpu().numpy() 


        #     axarr[0,n].imshow((out_data * 255).astype(np.uint8))
        #     axarr[1,n].imshow((test_data * 255).astype(np.uint8))
        #     n+=1
        # plt.show()



    


