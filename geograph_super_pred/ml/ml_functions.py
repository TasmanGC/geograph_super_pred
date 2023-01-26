import  torch
import  torch.nn as nn
from    sklearn.metrics import accuracy_score
from ..imports.import_classes import *

def nwMSE(pred_dict, true_dict, ntype_weights={'B':1,'A':2}):

    losses = []
    for key, value in pred_dict.items():
        if value.shape[0]!=0:
            loss = ntype_weights[key]*torch.mean((value - true_dict[key])**2)
            losses.append(loss)
    combined_loss = torch.stack(losses).mean()
    return combined_loss

def nwNLL(pred_dict, true_dict, ntype_weights={'B':1,'A':1},device='cpu'):
    
    loss_fn = nn.NLLLoss().to(device)
    m = nn.LogSoftmax(dim=1)
    # update true_dict
    losses = []
    for ntype, pred in pred_dict.items():
        if pred.shape[0]!=0:

            pred = m(pred)
            target  = true_dict[ntype].to(dtype=torch.int64)

            loss = loss_fn(pred,target) * ntype_weights[ntype] 
            losses.append(loss)
    combined_loss = torch.stack(losses).mean()
    return combined_loss

def accuracy_calc(pred_dict,true_dict):
    # TODO this might need to be updated, to reflect variable inputs dict for het, torch.Tensor for Homo

    acc_dict = {}
    acc_dict['A'] = 0
    acc_dict['B'] = 0
    for k,v in pred_dict.items():
        if v.shape[0]!=0:
            if v.shape[1]==2:
                prediction = torch.argmax(v,dim=1)
                true_value = true_dict[k].flatten()
                acc        = accuracy_score(true_value,prediction)
            if v.shape[1]==1:
                pass

            acc_dict[k] = acc

    return(acc_dict)

# psuedo train and test functions 
def homo_model_train(MODEL,TRAIN_LOADER,LOSS,M,OPTIM,in_size):

    for batch_idx, (_, _, mfgs) in enumerate(TRAIN_LOADER): # iterate over our training dataloader
    
        input = mfgs[0].srcdata['data'].reshape(-1,in_size)
        label = mfgs[-1].dstdata['labl'].to(dtype=torch.int64)
        
        predi = MODEL(mfgs, input)
        predi = M(predi)

        LOSS_V     = LOSS(predi,label.flatten())
        OPTIM.zero_grad()
        LOSS_V.backward()
        OPTIM.step()
    return(MODEL)

# make predictions
def homo_model_predict(model,dataloader,in_size,ntype,device=torch.device('cpu')):
    counter         = 0
    core_node_list  = []
    
    with torch.no_grad():
        for i,(_, _, mfgs) in enumerate(dataloader):
            input_mfg   = [g.to(device) for g in mfgs] 
            input       = input_mfg[0].srcdata['data'].reshape(-1,in_size)

            # make prediction extract info from the MFGS
            pred_dict   = model(input_mfg,input)

            labl_dict   = input_mfg[-1].ndata['labl']['_N']
            nloc_dict   = input_mfg[-1].ndata['nloc']['_N']

            for i,v in enumerate(pred_dict):

                pred = torch.argmax(pred_dict,dim=1)[i].detach().numpy()
                nloc = nloc_dict[i][0].detach().numpy()
                labl = labl_dict[i][0].detach().numpy()
                core_node_list.append(CoreNode(counter,nloc,{'pred':pred,'labl':labl,'type':ntype}))
                counter = counter+1
    return(core_node_list)

def hetero_model_train(MODEL,TRAIN_LOADER,OPTIM,device):

    for _, _, mfgs in TRAIN_LOADER: # iterate over the message flow graphs in our training dataloader

        OPTIM.zero_grad()

        # send the data to the device
        pred_dict   = MODEL(mfgs)
        true_dict   = mfgs[-1].dstdata['labl']

        # calculate loss
        LOSSV = nwNLL(pred_dict,true_dict,device=device)

        # back propigate and get set for next batch
        LOSSV.backward()
        OPTIM.step()
    return(MODEL)

# make predictions
def hetero_model_predict(model,dataloader,device=torch.device('cpu')):
        # move to cpu and test
    model.eval()
    model = model.to(device)

    counter         = 0
    core_node_list  = []
    
    with torch.no_grad():
        for i,(_, _, mfgs) in enumerate(dataloader):
            input_mfg   = [g.to(device) for g in mfgs] 

            # make prediction extract info from the MFGS
            pred_dict   = model(input_mfg)
            labl_dict   = input_mfg[-1].ndata['labl']
            nloc_dict   = input_mfg[-1].ndata['nloc']

            # make a set of core nodes for that batch
            for key, value in pred_dict.items():
                for i,v in enumerate(value):
                    pred = torch.argmax(pred_dict[key],dim=1)[i].detach().numpy()
                    nloc = nloc_dict[key][i].detach().numpy()
                    labl = labl_dict[key][i].detach().numpy()
                    core_node_list.append(CoreNode(nloc,{'pred':pred,'labl':labl,'type':key}))
                    counter = counter+1
    return(core_node_list)