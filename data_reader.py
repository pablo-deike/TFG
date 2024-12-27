import glob 
import re 
import numpy as np
import pandas as pd

csv_files = glob.glob('C:/Users/pablo/Desktop/scripts tfg/archivos min/*.min')

combined_df_min = pd.DataFrame()
ind=['min HFB','min PNP','min PNPAMP']
combined_df_min.index=ind
difE=[]
list_zn=[]
E_HFB=[]
E_PNP=[]
E_PNPAMP=[]
for csv_file in csv_files:
    df = pd.read_csv(csv_file,sep=' ',header=None)
    df.columns=['beta_2','E HFB','E PNP', 'E PNPAMP']
    zn=[int(s) for s in re.findall(r'\d+', csv_file)] 
    list_zn.append(zn)
    E_hfb=df.iloc[0,1]
    E_pnp=df.iloc[1,2]
    E_pnpamp=df.iloc[2,3]
    difE.append(E_pnpamp-E_hfb)
    E_HFB.append(E_hfb)
    E_PNPAMP.append(E_pnpamp)
    E_PNP.append(E_pnp)

np.savetxt('data_E.csv', pd.concat([pd.DataFrame(list_zn), pd.DataFrame(difE), pd.DataFrame(E_HFB), pd.DataFrame(E_PNP), pd.DataFrame(E_PNPAMP)], axis=1, join='inner'), header='Z,N,difE,E_HFB,E_PNP,E_PNPAMP', delimiter=',')
