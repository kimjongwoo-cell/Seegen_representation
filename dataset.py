from torch.utils.data import Dataset
import cv2
import pandas as pd




class NMDataset(Dataset):

    def __init__(self,split,transform=None):
        super().__init__()
        self.path = r"D:/NewEra"
        self.transform = transform
        self.split = split
        self.df_N = pd.read_csv(self.path + '/N_'+split+'.csv')
        self.df_M = pd.read_csv(self.path + '/M_'+split+'.csv')
        self.df_no = pd.read_csv(self.path + '/No_'+split+'.csv')
        self.df = pd.concat([self.df_M,self.df_N,self.df_no],ignore_index=True) # 행 번호를 다시 0부터 쭈욱
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        del(self.df_N); del(self.df_M); del(self.df_no)

    def __len__(self):
        return len(self.df)

    def get_img(self,pat_id,file_id,label_id,patch_id=None):
        if label_id == 'M':
            image = cv2.imread("{}/M/{}_{}.png".format(self.path, pat_id, file_id), cv2.IMREAD_COLOR)
            image = self.transform(image)
            label = 0
            return image, label

        elif label_id == "N":
            image = cv2.imread("{}/N/{}_{}.png".format(self.path , pat_id, file_id), cv2.IMREAD_COLOR)
            image= self.transform(image)
            label = 1
            return image, label

        else :
            image = cv2.imread("{}/No/{}_{}.png".format(self.path , pat_id, file_id), cv2.IMREAD_COLOR)
            image= self.transform(image)
            label = 2
            return image, label
        

    def __getitem__(self, idx):
        pat_id, file_id, label_id = self.df.iloc[idx,:]
        images, labels = self.get_img(pat_id, file_id,label_id)

        return images, labels