from turtle import forward
import torch


# model prepare



a= 1
def Training(model,Loss1,Loss2, optimizer,train_loader):
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    try:
        for i, (data, label) in enumerate(train_loader):
            img = data.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            # ===================forward=====================
            conf_level,output_encoder,output_decoder= model(img)

            conf_level.requires_grad_(True)
            output_encoder.requires_grad_(True)
            output_decoder.requires_grad_(True)


            loss1 = Loss1(conf_level, label)   # classifier 예측
            loss2 = Loss2(img,output_decoder) # autoencoder 예측
            loss = loss1+loss2
            # ===================backward====================
            
            loss.backward()
            optimizer.step()
            total_loss += loss.data
            total_loss1 += loss1.data
            total_loss2 += loss2.data

            
    except TypeError: 
        pass       
    # ===================log========================
    


    return total_loss/(i+1), total_loss1/(i+1),total_loss2/(i+1)

from sklearn.metrics import f1_score

def validate(model,Loss1,Loss2, valid_loader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        f1 = 0
        pred =[]
        real =[]
        try:
            for i, (data, label) in enumerate(valid_loader):
                img = data.cuda()
                label = label.cuda()

                # ===================forward=====================
                conf_level,output_encoder,output_decoder= model(img)

                
                loss1 = Loss1(conf_level, label)   # classifier 예측
                loss2 = Loss2(img,output_decoder) # autoencoder 예측
                   # 가중인자 a 추가
                loss = loss1+loss2
                total_loss += loss.data
                ##total_loss2 += loss2.data





                pred += conf_level.argmax(dim=1).cpu().tolist()
                real+= label.cpu().numpy().tolist()
                

        # ===================log========================
        except TypeError: 
            pass     
    return total_loss/(i+1), f1_score(real, pred, average='micro')




