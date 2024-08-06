import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#df= pd.read_csv("/home/deike/data/data_E.csv")
df= pd.read_csv("/Users/pablo/Desktop/scripts tfg/data_E.csv")
list_zn=df[['# Z','N']].values.tolist()
difE=df['difE'].values.tolist()
x=torch.tensor(list_zn)
y=torch.tensor(np.abs(difE))


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

def create_dataloaders(x, y, split_ratio, batch_size):
    dataset = TensorDataset(x, y)
    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=test_size, shuffle=False)

    return train_loader, test_loader, train_size

def training_model(x,y,data_size, learning_rate, batch_size): 
    train_loader, test_loader, train_size = create_dataloaders(x,y,split_ratio=data_size, batch_size=batch_size)
    net_slice=MyModel(2,1)
    net_slice=net_slice.float()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net_slice.parameters(), lr=learning_rate)        
    dloss=1
    i=0 
    nepoch=0
    net_slice.train()
    hatY=torch.zeros(train_size,1)
    n_batches=int(np.floor(train_size/batch_size))
    new_loss=np.zeros(n_batches)
    previous_loss=np.zeros(n_batches)
    print(f"training with {n_batches} batches in {train_size} points of data with learning_rate={learning_rate}")
    while dloss > 1e-2 and nepoch<5000:
        for batch_idx, (x_train_slice, y_train_slice) in enumerate(train_loader):
            x_train_slice = x_train_slice.float()
            y_train_slice = y_train_slice.float()
            hatY_slice = net_slice.forward(x_train_slice)
            optimizer.zero_grad()
            loss_slice = criterion(hatY_slice, y_train_slice)
            loss_slice.backward()
            optimizer.step()
            #x_train.append(x_train_slice)
            #y_train.append(y_train_slice)
            if nepoch % 50 == 0:
                new_loss[batch_idx]=loss_slice.item()
                print(f"The loss function for {nepoch} is {new_loss} and batch numb={batch_idx}")
                if batch_idx==n_batches-1:
                    dloss=np.abs(np.mean(new_loss-previous_loss))
                    print(f"current diff{dloss}")
                    previous_loss=new_loss
                    new_loss=np.zeros(n_batches)
        nepoch+=1
    if dloss<1e-2:
        has_finished=True
    else: 
        has_finished=False
    all_x_train = torch.Tensor()
    all_y_train = torch.Tensor()
    for x_train, y_train in train_loader:
        all_x_train = torch.cat((all_x_train, x_train), 0)
        all_y_train = torch.cat((all_y_train, y_train), 0)
    hatY=net_slice.forward(all_x_train)
    #elapsed_time = time.time() - start_time
    with torch.no_grad():
        net_slice.eval()
        for x_test, y_test in test_loader:
            x_test=x_test.float()
            y_test=y_test.float()
            hatY_pred=net_slice.forward(x_test)
            hatY_detach=hatY_pred.detach()
            test_loss=criterion(hatY_detach,y_test)


    return test_loss, loss_slice, nepoch, hatY, hatY_pred, all_x_train, x_test, all_y_train, y_test, has_finished


#grid search
learning_rates=np.logspace(-5,-1,5)
data_sizes=np.array([0.3,0.4,0.5,0.6,0.7,0.8])
test_loss=np.zeros((len(data_sizes), len(learning_rates)))
train_loss= np.zeros_like(test_loss)
nepochs=np.zeros_like(test_loss)
has_finished_mat=np.zeros_like(test_loss)
fontsize=16
for a_iter, data_size in enumerate(data_sizes):
    for b_iter, learning_rate in enumerate(learning_rates):
        total_loss_slice, loss_slice, nepoch, hatY_slice, hatY_pred_slice,  x_train, x_test_slice, y_train, y_test_slice, has_finished=training_model(x,y,data_size=data_size, learning_rate=learning_rate, batch_size=128)
        test_loss[a_iter,b_iter]=total_loss_slice.item()
        train_loss[a_iter, b_iter]=loss_slice.item()
        nepochs[a_iter,b_iter]=nepoch
        has_finished_mat[a_iter,b_iter]=has_finished


df = pd.DataFrame(test_loss, 
                  columns=['1e-05', '0.0001', '0.001', '0.01', '0.1'],  # Example learning rates
                  index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # Example data sizes
df.to_csv('Adam_batches_test_loss.csv')

df = pd.DataFrame(train_loss, 
                  columns=['1e-05', '0.0001', '0.001', '0.01', '0.1'],  # Example learning rates
                  index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # Example data sizes
df.to_csv('Adam_batches_train_loss.csv')

df = pd.DataFrame(nepochs, 
                  columns=['1e-05', '0.0001', '0.001', '0.01', '0.1'],  # Example learning rates
                  index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # Example data sizes
df.to_csv('Adam_batches_nepochs.csv')

df = pd.DataFrame(has_finished_mat,
                    columns=['1e-05', '0.0001', '0.001', '0.01', '0.1'],  # Example learning rates
                  index=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # Example data sizes
df.to_csv('Adam_batches_has_finished.csv')

