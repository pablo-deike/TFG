import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#df= pd.read_csv('data_E.csv')
df= pd.read_csv('/home/deike/data/data_E.csv')
zn_E=df[['# Z','N','difE','E_HFB']].values.tolist()
difE=df['E_PNPAMP'].values.tolist()



class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 750) 
        self.fc3 = nn.Linear(750,600)
        self.fc4 = nn.Linear (600, 800)
        self.fc5 = nn.Linear(800, output_dim)
        self.internal = nn.ReLU()
        self.activation = nn.ReLU()
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)        
        self.batchnorm1=nn.BatchNorm1d(1000, eps=1e-05, momentum=None)#batch norm should be frozen when evaluating the model
        self.batchnorm2=nn.BatchNorm1d(750, eps=1e-05, momentum=None)
        self.batchnorm3=nn.BatchNorm1d(600, eps=1e-05, momentum=None)
        self.batchnorm4=nn.BatchNorm1d(800, eps=1e-05, momentum=None)

    def forward(self, x):
        x1 = self.internal(self.fc1(x))
        x1= self.batchnorm1(x1)
        x1= self.dropout(x1)
        x2 = self.internal(self.fc2(x1))
        x2=self.batchnorm2(x2)
        x2= self.dropout(x2)
        x3 = self.internal(self.fc3(x2))
        x3=self.batchnorm3(x3)
        x3= self.dropout(x3)
        x4 = self.internal(self.fc4(x3))
        x4=self.batchnorm4(x4)
        x4= self.dropout(x4)
        y=self.activation(self.fc5(x4))
        return y 


from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(x, y, split_ratio, batch):
    dataset = TensorDataset(x, y)

    train_size = int(len(dataset) * split_ratio/batch)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    batch_size=int(train_size/batch)
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=test_size, shuffle=False)

    return train_loader, test_loader


def training_model(x,y,data_size, learning_rate, batch_size): 
    train_loader, test_loader = create_dataloaders(x,y,split_ratio=data_size, batch=batch_size)
    net_slice=MyModel(4,1)
    net_slice=net_slice.float()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(net_slice.parameters(), lr=learning_rate)
    #start_time=time.time()        
    dloss=1
    loss_item=[]
    loss_item.append(0.0)
    i=0 
    nepoch=0
    net_slice.train()
    while dloss > 1e-2 :
        for batch_idx, (x_train_slice, y_train_slice) in enumerate(train_loader):
            x_train_slice = x_train_slice.float()
            y_train_slice = y_train_slice.float()
            hatY_slice = net_slice.forward(x_train_slice)
            optimizer.zero_grad()
            loss_slice = criterion(hatY_slice, y_train_slice)
            loss_slice.backward()
            optimizer.step()
            # solo sobre el test    
            #acc=np.std()#loss segun tamaño entreno y test y varianza
            #entrenar sobre las propias energías
            #hablar sobre tiempos de computación
            #rel_acc=acc/y_test.mean()
            if nepoch % 100 == 0 and batch_idx==batch_size-1:
                loss_item.append(loss_slice.item())
                dloss=np.abs(loss_item[i+1]-loss_item[i])
                i+=1
        nepoch+=1
            
    #elapsed_time = time.time() - start_time
    net_slice.eval()
    for x_test_slice, y_test_slice in test_loader:
        x_test_slice=x_test_slice.float()
        y_test_slice=y_test_slice.float()
        hatY_pred_slice=net_slice.forward(x_test_slice)
        hatY_detach_slice=hatY_pred_slice.detach()
        total_loss_slice=criterion(hatY_detach_slice,y_test_slice)
    return total_loss_slice, dloss, loss_slice, nepoch, hatY_slice, hatY_pred_slice, x_train_slice, x_test_slice, y_train_slice, y_test_slice

#grid search
x=torch.tensor(zn_E)
y=torch.tensor(np.abs(difE))
learning_rates=np.logspace(-5,-1,5)
data_sizes=np.array([0.3,0.4,0.5,0.6,0.7,0.8])
test_loss=np.zeros((len(data_sizes), len(learning_rates)))
train_loss= np.zeros_like(test_loss)
nepochs=np.zeros_like(test_loss)
fontsize=16
for a_iter, data_size in enumerate(data_sizes):
    for b_iter, learning_rate in enumerate(learning_rates):
        total_loss_slice, dloss, loss_slice, i, hatY_slice, hatY_pred_slice,  x_train_slice, x_test_slice, y_train_slice, y_test_slice=training_model(x=x,y=y,data_size=data_size, learning_rate=learning_rate, batch_size=4)
        test_loss[a_iter,b_iter]=total_loss_slice.item()
        train_loss[a_iter, b_iter]=loss_slice.item()
        nepochs[a_iter,b_iter]=i


df = pd.DataFrame(test_loss, 
                  columns=['1e-05', '0.0001', '0.001', '0.01', '0.1'],  # Example learning rates
                  index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # Example data sizes

# Save to CSV
df.to_csv('E_pnpamp_test_loss.csv')
df = pd.DataFrame(train_loss, 
                  columns=['1e-05', '0.0001', '0.001', '0.01', '0.1'],  # Example learning rates
                  index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # Example data sizes

# Save to CSV
df.to_csv('E_pnpamp_train_loss.csv')
df = pd.DataFrame(nepochs, 
                  columns=['1e-05', '0.0001', '0.001', '0.01', '0.1'],  # Example learning rates
                  index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # Example data sizes

# Save to CSV
df.to_csv('E_pnpamp_nepochs.csv')
