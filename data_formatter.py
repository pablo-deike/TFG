import glob
import re
import pandas as pd
import numpy as np

csv_files = glob.glob("data/files_min/*.min")

# Create an empty dataframe to store the combined data
combined_df_min = pd.DataFrame()
ind = ["min HFB", "min PNP", "min PNPAMP"]
combined_df_min.index = ind
E_HFB = []
E_PNPAMP = []
E_PNP = []
list_zn = []
# Loop through each CSV file and append its contents to the combined dataframe
for csv_file in csv_files:
    df = pd.read_csv(csv_file, sep=" ", header=None)
    df.columns = ["beta_2", "E HFB", "E PNP", "E PNPAMP"]
    zn = [int(s) for s in re.findall(r"\d+", csv_file)]
    list_zn.append(zn)
    E_HFB.append(np.abs(df.iloc[0, 1]))
    E_PNP.append(np.abs(df.iloc[1, 2]))
    E_PNPAMP.append(np.abs(df.iloc[2, 3]))
parsed_data = pd.DataFrame(
    {"zn": list_zn, "E_HFB": E_HFB, "E_PNP": E_PNP, "E_PNPAMP": E_PNPAMP}
)
parsed_data.to_csv("data/energies.csv")
