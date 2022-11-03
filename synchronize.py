from tqdm import tqdm
import os
def synchronized(path, data_type=None):
    '''
    path : csv파일과 이미지 폴더가 있는 dir
    '''
    assert data_type in ['N','M',"No"]
    preprocess_path = path
    if data_type == 'M':
        file_list = next(os.walk(preprocess_path+'M/'))[2]
        M_train = pd.read_csv(preprocess_path + 'M_train.csv')
        M_val = pd.read_csv(preprocess_path + 'M_val.csv')
        M_test = pd.read_csv(preprocess_path + 'M_test.csv')
        # 각자 df돌며 해당하는 이미지와 마스크가 없다면 df에서 삭제
        copy_M_train = pd.DataFrame(columns=["pat_id", "file_id"])
        copy_M_val = pd.DataFrame(columns=["pat_id", "file_id"])
        copy_M_test = pd.DataFrame(columns=["pat_id", "file_id"])
        for i in tqdm(range(len(M_train)),desc='M train data 확인중...'):
            pat_id, file_id,label = M_train.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
            if file_name in file_list:
                copy_M_train = copy_M_train.append({'pat_id':pat_id,'file_id':file_id,'label_id':label}, ignore_index=True)
        for i in tqdm(range(len(M_val)),desc='M val data 확인중...'):
            pat_id, file_id,label = M_val.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
            if file_name in file_list:
                copy_M_val = copy_M_val.append({'pat_id':pat_id,'file_id':file_id,'label_id':label}, ignore_index=True)
        for i in tqdm(range(len(M_test)),desc='M test data 확인중...'):
            pat_id, file_id,label = M_test.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
            if file_name in file_list:
                copy_M_test = copy_M_test.append({'pat_id':pat_id,'file_id':file_id,'label_id':label}, ignore_index=True)
        del(M_train)
        del(M_val)
        del(M_test)
        copy_M_train.to_csv(preprocess_path + 'new_M_train.csv', index=False)
        copy_M_val.to_csv(preprocess_path + 'new_M_val.csv', index=False)
        copy_M_test.to_csv(preprocess_path + 'new_M_test.csv', index=False)
    elif data_type == 'N':
        file_list = next(os.walk(preprocess_path+'N/'))[2]
        N_train = pd.read_csv(preprocess_path + 'N_train.csv')
        N_val = pd.read_csv(preprocess_path + 'N_val.csv')
        N_test = pd.read_csv(preprocess_path + 'N_test.csv')
        # 각자 df돌며 해당하는 이미지와 마스크가 없다면 df에서 삭제
        copy_N_train = pd.DataFrame(columns=["pat_id", "file_id"])
        copy_N_val = pd.DataFrame(columns=["pat_id", "file_id"])
        copy_N_test = pd.DataFrame(columns=["pat_id", "file_id"])
        for i in tqdm(range(len(N_train)),desc='N train data 확인중...'):
            pat_id, file_id = N_train.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
            if file_name in file_list:
                copy_N_train = copy_N_train.append({'pat_id':pat_id,'file_id':file_id}, ignore_index=True)
        for i in tqdm(range(len(N_val)),desc='N val data 확인중...'):
            pat_id, file_id = N_val.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
            if file_name in file_list:
                copy_N_val = copy_N_val.append({'pat_id':pat_id,'file_id':file_id}, ignore_index=True)
        for i in tqdm(range(len(N_test)),desc='N test data 확인중...'):
            pat_id, file_id= N_test.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
            if file_name in file_list:
                copy_N_test = copy_N_test.append({'pat_id':pat_id,'file_id':file_id}, ignore_index=True)
        del(N_train)
        del(N_val)
        del(N_test)
        copy_N_train.to_csv(preprocess_path + 'new_N_train.csv', index=False)
        copy_N_val.to_csv(preprocess_path + 'new_N_val.csv', index=False)
        copy_N_test.to_csv(preprocess_path + 'new_N_test.csv', index=False)
    elif data_type == 'No':
        file_list = next(os.walk(preprocess_path+'No/'))[2]
        M_train = pd.read_csv(preprocess_path + 'No_train.csv')
        M_val = pd.read_csv(preprocess_path + 'No_val.csv')
        M_test = pd.read_csv(preprocess_path + 'No_test.csv')
        # 각자 df돌며 해당하는 이미지와 마스크가 없다면 df에서 삭제
        copy_M_train = pd.DataFrame(columns=["pat_id", "file_id",'label_id'])
        copy_M_val = pd.DataFrame(columns=["pat_id", "file_id",'label_id'])
        copy_M_test = pd.DataFrame(columns=["pat_id", "file_id",'label_id'])
        for i in tqdm(range(len(M_train)),desc='No train data 확인중...'):
            pat_id, file_id,label = M_train.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
            if file_name in file_list:
                copy_M_train = copy_M_train.append({'pat_id':pat_id,'file_id':file_id,'label_id':label}, ignore_index=True)
        for i in tqdm(range(len(M_val)),desc='No val data 확인중...'):
            pat_id, file_id,label = M_val.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
            
            if file_name in file_list:
                print(file_name)
                copy_M_val = copy_M_val.append({'pat_id':pat_id,'file_id':file_id,'label_id':label}, ignore_index=True)
        for i in tqdm(range(len(M_test)),desc='No test data 확인중...'):
            pat_id, file_id,label = M_test.loc[i].values
            file_name = pat_id + '_' + str(file_id) + '.png'
          
            if file_name in file_list:
                copy_M_test = copy_M_test.append({'pat_id':pat_id,'file_id':file_id,'label_id':label}, ignore_index=True)
        del(M_train)
        del(M_val)
        del(M_test)
        copy_M_train.to_csv(preprocess_path + 'new_No_train.csv', index=False)
        copy_M_val.to_csv(preprocess_path + 'new_No_val.csv', index=False)
        copy_M_test.to_csv(preprocess_path + 'new_No_test.csv', index=False)



synchronized("D:/NewEra/",'No')