
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split


def get_train_dev_test_data(all_data):
    train, test = train_test_split(all_data, test_size=0.2, random_state=42)
    train, dev = train_test_split(train, test_size=0.1, random_state=42)
    return train, dev, test

def show_image(outputs,tests):
    f, axarr = plt.subplots(2,5,figsize= (10,8))
    n = 0
    for out_data, test_data in zip(outputs,tests):
        out_data = out_data.permute(1,2,0).cpu().detach().numpy()   
        test_data  = test_data.permute(1,2,0).cpu().numpy() 


        axarr[0,n].imshow((out_data * 255).astype(np.uint8))
        axarr[1,n].imshow((test_data * 255).astype(np.uint8))
        n+=1
    plt.show()