# data split model
#train/test/val로 나눔
path="D:/NewEra/M"
A=[]

for i in os.listdir("D:/NewEra/M"):
    file=i.split("_")

    if len(i.split("_"))==3:
        A.append(file[0]+"_1")
    else:
        A.append(file[0])
B=list(set(A))
print(len(B))
train_folder = B[:28]
val_folder = B[28:35]
test_folder = B[35:]


train_file=[]
train_folders=[]
train_label=[]
test_file=[]
test_folders=[]
test_label=[]
val_file=[]
val_folders=[]
val_label=[]
no=[]
for file in os.listdir(path):

        if len(file.split("_"))==3:
            folder_name = file.split("_")[0]+"_1"
        else:
            folder_name = file.split("_")[0]
        
        if folder_name in val_folder:
            val_file.append([folder_name,file.split("_")[-1][:-4],"M"])
            # val_folders.append(folder_name)
            # val_label.append("No")
        elif folder_name in test_folder:
            test_file.append([folder_name,file.split("_")[-1][:-4],"M"])
            # test_folders.append(folder_name)
            # test_label.append("No")
        elif folder_name in train_folder:
            train_file.append([folder_name,file.split("_")[-1][:-4],"M"])
            # train_folders.append(folder_name)
            # train_label.append("No")
        else:
            print(folder_name)
        
        no.append(folder_name)


train_df =pd.DataFrame(train_file,columns=["pat_id","file_id","label_id"])
val_df = pd.DataFrame(val_file,columns=["pat_id","file_id","label_id"])
test_df = pd.DataFrame(test_file,columns=["pat_id","file_id","label_id"])
train_df.to_csv("D:/NewEra/" + 'M_train.csv', index=False)
val_df.to_csv("D:/NewEra/" + 'M_val.csv', index=False)
test_df.to_csv("D:/NewEra/" + 'M_test.csv', index=False)