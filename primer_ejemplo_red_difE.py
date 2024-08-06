import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import glob 
import re

#importar archivos, tarda tiempo una vez se importan mejor guardalos
csv_files = glob.glob('C:/Users/pablo/Desktop/scripts tfg/archivos min/*.min')

# Create an empty dataframe to store the combined data
combined_df_min = pd.DataFrame()
ind=['min HFB','min PNP','min PNPAMP']
combined_df_min.index=ind
difE=[]
list_zn=[]
# Loop through each CSV file and append its contents to the combined dataframe
for csv_file in csv_files:
    df = pd.read_csv(csv_file,sep=' ',header=None)
    df.columns=['beta_2','E HFB','E PNP', 'E PNPAMP']
    zn=[int(s) for s in re.findall(r'\d+', csv_file)] 
    list_zn.append(zn)
    E_hfb=df.iloc[0,1]
    E_pnpamp=df.iloc[2,3]
    difE.append(E_pnpamp-E_hfb)
x=torch.tensor(list_zn)
y=torch.tensor(np.abs(difE))

#red neuronal
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 1750) 
        self.fc3 = nn.Linear(1750, output_dim)
        self.activation=nn.ReLU()
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)        
        self.batchnorm1=nn.BatchNorm1d(1000, eps=1e-05, momentum=None)
        self.batchnorm2=nn.BatchNorm1d(1750, eps=1e-05, momentum=None)
    def forward(self, x):
        x1 = self.activation(self.fc1(x))
        x1= self.batchnorm1(x1)
        x2 = self.activation(self.fc2(x1))
        x2=self.batchnorm2(x2)
        y=self.activation(self.fc3(x2))
        return y 
    
#entrenamiento
x_train_slice=x[0::2, :].float()
y_train_slice=y[0::2].float()
net_slice=MyModel(2,1)
y_test_slice=y[1::2].float()
x_test_slice=x[1::2, :].float()
net_slice=net_slice.float()
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net_slice.parameters(), lr=1e-2)
for nepoch in range(2000):
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
    if nepoch % 100 == 0:
        print(f"loss for {nepoch} = {loss_slice.item():0.2f}")

hatY_pred_slice=net_slice.forward(x_test_slice)
hatY_detach_slice=hatY_pred_slice.detach()
total_loss_slice=criterion(hatY_detach_slice,y_test_slice)
print(total_loss_slice.item())