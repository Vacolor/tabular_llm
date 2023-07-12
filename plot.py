import torch
from torch import nn
from models import BertBased

from data_openml import data_prep_openml,task_dset_ids,DataSetCatCon
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, classification_scores, mean_sq_error
from augmentations import embed_data_mask

import os
import numpy as np



parser = argparse.ArgumentParser()

parser.add_argument('--dset_id', required=True, type=int)
parser.add_argument('--vision_dset', action = 'store_true')
parser.add_argument('--task', required=True, type=str, choices = ['binary','multiclass','regression'])
parser.add_argument('--cont_embeddings', default='MLP', type=str, choices = ['MLP','Noemb','pos_singleMLP'])

parser.add_argument('--optimizer', default='AdamW', type=str, choices = ['AdamW','Adam','SGD'])
parser.add_argument('--scheduler', default='cosine', type=str, choices = ['cosine','linear'])

parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batchsize', default=256, type=int)
parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
parser.add_argument('--run_name', default='altered_saint', type=str) # group name for wandb
parser.add_argument('--set_seed', default= 1 , type=int)
parser.add_argument('--dset_seed', default= 5 , type=int)
parser.add_argument('--active_log', action = 'store_true') # must on for plotting

parser.add_argument('--train_mask_prob', default=0, type=float) # probability
parser.add_argument('--mask_prob', default=0, type=float)

parser.add_argument('--ssl_avail_y', default= 0, type=int)
parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
parser.add_argument('--nce_temp', default=0.7, type=float)

parser.add_argument('--lam0', default=0.5, type=float)
parser.add_argument('--lam1', default=10, type=float)
parser.add_argument('--lam2', default=1, type=float)
parser.add_argument('--lam3', default=10, type=float)
parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])

parser.add_argument('--backbone', default = 'bert-base-uncased', type = str)
parser.add_argument('--accelerate', action = 'store_true')
parser.add_argument('--step', default = 5, type = int) # run test set when epoch%step == 0

opt = parser.parse_args()
modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.task,str(opt.dset_id),opt.run_name)
if opt.task == 'regression':
    opt.dtask = 'reg'
else:
    opt.dtask = 'clf'

torch.manual_seed(opt.set_seed)
os.makedirs(modelsave_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")
#device = torch.device("cpu")

torch.cuda.empty_cache()



##################creating wandb project##################
if opt.active_log:
    import wandb
    """
    if opt.pretrain:
        wandb.init(project="saint_v2_all", group =opt.run_name ,name = f'pretrain_{opt.task}_{str(opt.attentiontype)}_{str(opt.dset_id)}_{str(opt.set_seed)}')
    else:"""
    run = wandb.init(project = "saint", group = opt.run_name, name = f'setid:_{str(opt.dset_id)}_task:_{opt.task}_seed={str(opt.set_seed)}')
    wandb.config.update(opt)
# it can be told here that argparse may be significant for utilizing wandb



##################preparing data##################
print('Downloading and processing the dataset, it might take some time.')
# cat_dims: dimensions of categories for each column
# cat_idxs: indices for categorial columns
cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep_openml(opt.dset_id, opt.dset_seed, opt.task, datasplit=[.65, .15, .2])
continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32) 

train_ds = DataSetCatCon(X_train, y_train, cat_idxs, opt.dtask, continuous_mean_std)
trainloader = DataLoader(train_ds, batch_size=opt.batchsize, shuffle=True, num_workers=4)

valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, opt.dtask, continuous_mean_std)
validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)

test_ds = DataSetCatCon(X_test, y_test, cat_idxs, opt.dtask, continuous_mean_std)
testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False, num_workers=4)
if opt.task == 'regression':
    y_dim = 1
else:
    y_dim = len(np.unique(y_train['data'][:,0]))

cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int) #Appending 1 for CLS token, this is later used to generate embeddings.



##################setting model##################
model = BertBased(
categories = tuple(cat_dims), 
num_continuous = len(con_idxs),                
dim_out = 1,                                      
mlp_hidden_mults = (4, 2),       
cont_embeddings = opt.cont_embeddings,
final_mlp_style = opt.final_mlp_style,
y_dim = y_dim
)
vision_dset = opt.vision_dset
    
# Loss function
if y_dim == 2 and opt.task == 'binary':
    # opt.task = 'binary'
    criterion = nn.CrossEntropyLoss().to(device)
elif y_dim > 2 and  opt.task == 'multiclass':
    # opt.task = 'multiclass'
    criterion = nn.CrossEntropyLoss().to(device)
elif opt.task == 'regression':
    criterion = nn.MSELoss().to(device)
else:
    raise'case not written yet'

model.to(device)

# Optimizer
if opt.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=0.9, weight_decay=5e-4)
    from utils import get_scheduler
    scheduler = get_scheduler(opt, optimizer)
elif opt.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
elif opt.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000

# accelerator from hugginface
if opt.accelerate:
    from accelerate import Accelerator    
    accelerator = Accelerator()
    trainloader, validloader, testloader, model, optimizer = accelerator.prepare(
        trainloader, validloader, testloader, model, optimizer
    )



##################training##################
print('Training begins now.')
n_batch = 0
for epoch in range(opt.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        '''
        # wandb watching
        if opt.active_log:
            wandb.watch(model, criterion, log = 'all', log_freq = 10)
        '''
        # get data
        # x_categ is the the categorical data, x_cont has continuous data, y_gts has ground truth ys. ## wrong ## cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s. con_mask is an array of ones same shape as x_cont. 
        # mask has 0s stands for missing values and 1s stands for not missing values
        x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

        # embedding tokens
        _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)           
        
        # passing transformer blocks
        reps = model(x_categ_enc, x_cont_enc) # shape: b, n, d

        # select CLS tokens and send to final mlp
        y_reps = reps[:, 0, :]
        y_outs = model.mlpfory(y_reps)#.requires_grad()
        
        # loss
        if opt.task == 'regression':
            loss = criterion(y_outs, y_gts) 
        else:
            loss = criterion(y_outs, y_gts.squeeze()) 
        #loss.backward()
        if opt.accelerate:
            accelerator.backward(loss)
        else:
            loss.backward()
        
        # optimize
        optimizer.step()
        if opt.optimizer == 'SGD':
            scheduler.step()
            
        running_loss += loss.item()
        
        # update loss to wandb per batch
        n_batch += 1
        if opt.active_log:
            wandb.log({'train_batch_loss': loss.item()}, step = n_batch)
        
    # update loss to wandb per epoch
    if opt.active_log:
        wandb.log({'epoch': epoch, 'train_epoch_loss': running_loss})

    # evaluating
    if (epoch+1) % opt.step == 0:
        model.eval()
        with torch.no_grad():
            if opt.task in ['binary', 'multiclass']:
                # deriving scores
                valid_accuracy, valid_auroc = classification_scores(model, validloader, device, opt.task, vision_dset)
                test_accuracy, test_auroc = classification_scores(model, testloader, device, opt.task, vision_dset)
                print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                    (epoch + 1, valid_accuracy, valid_auroc))
                print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                    (epoch + 1, test_accuracy, test_auroc))
                
                # deriving losses
                valid_loss = 0
                for i, data in enumerate(validloader, 0):
                    x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
                    _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
                    reps = model(x_categ_enc, x_cont_enc) # shape: b, n, d
                    y_reps = reps[:, 0, :]
                    y_outs = model.mlpfory(y_reps)#.requires_grad()
                    valid_loss += criterion(y_outs, y_gts.squeeze())
                    
                test_loss = 0
                for i, data in enumerate(testloader, 0):
                    x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
                    _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
                    reps = model(x_categ_enc, x_cont_enc) # shape: b, n, d
                    y_reps = reps[:, 0, :]
                    y_outs = model.mlpfory(y_reps)#.requires_grad()
                    test_loss += criterion(y_outs, y_gts.squeeze())
                    
                # log to wandb
                if opt.active_log:
                    wandb.log({'valid_accuracy': valid_accuracy, 'test_accuracy': test_accuracy, 'valid_loss': valid_loss, 'test_loss': test_loss})
                    if opt.task == 'binary':
                        wandb.log({'valid_auroc': valid_auroc, 'test_auroc': test_auroc})
                
                # save the bests as checkpoints
                if opt.task == 'multiclass':
                    if valid_accuracy > best_valid_accuracy:
                        best_valid_accuracy = valid_accuracy
                        best_test_accuracy = test_accuracy
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                else:
                    if valid_accuracy > best_valid_accuracy:
                        best_valid_accuracy = valid_accuracy
                    # if auroc > best_valid_auroc:
                    #     best_valid_auroc = auroc
                        best_test_auroc = test_auroc
                        best_test_accuracy = test_accuracy               
                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))

            else: # regression task
                valid_rmse = mean_sq_error(model, validloader, device,vision_dset)    
                test_rmse = mean_sq_error(model, testloader, device,vision_dset)  
                print('[EPOCH %d] VALID RMSE: %.3f' %
                    (epoch + 1, valid_rmse))
                print('[EPOCH %d] TEST RMSE: %.3f' %
                    (epoch + 1, test_rmse))
                
                # deriving losses
                valid_loss = 0
                for i, data in enumerate(validloader, 0):
                    x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
                    _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
                    reps = model(x_categ_enc, x_cont_enc) # shape: b, n, d
                    y_reps = reps[:, 0, :]
                    y_outs = model.mlpfory(y_reps)
                    valid_loss += criterion(y_outs, y_gts)
                    
                test_loss = 0
                for i, data in enumerate(testloader, 0):
                    x_categ, x_cont, y_gts, cat_mask, con_mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
                    _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset)
                    reps = model(x_categ_enc, x_cont_enc) # shape: b, n, d
                    y_reps = reps[:, 0, :]
                    y_outs = model.mlpfory(y_reps)
                    test_loss += criterion(y_outs, y_gts)
                
                if opt.active_log:
                    wandb.log({'valid_rmse': valid_rmse, 'test_rmse': test_rmse, 'valid_loss': valid_loss, 'test_loss': test_loss})     
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_test_rmse = test_rmse
                    torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
    
    # end of evaluating
        model.train()
                


##################sending conclusions##################
total_parameters = count_parameters(model)
print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
if opt.task =='binary':
    print('AUROC on best model:  %.3f' %(best_test_auroc))
elif opt.task =='multiclass':
    print('Accuracy on best model:  %.3f' %(best_test_accuracy))
else:
    print('RMSE on best model:  %.3f' %(best_test_rmse))

if opt.active_log:
    if opt.task == 'regression':
        wandb.log({'total_parameters': total_parameters, 'test_rmse_bestep': best_test_rmse, 
        'cat_dims': len(cat_idxs), 'con_dims':len(con_idxs) })        
    else: 
        wandb.log({'total_parameters': total_parameters,
        'test_accuracy_bestep': best_test_accuracy, 'cat_dims': len(cat_idxs), 'con_dims': len(con_idxs) })
        if opt.task == 'binary':
            wandb.log({'test_auroc_bestep': best_test_auroc})
