import os
from utils.load_mat import loadmat

split = loadmat("../SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat") 

N_train = len(split["alltrain"])
N_val = len(split["alltest"])

hash_train = []
for i in range(N_train):
    folder_path = split["alltrain"][i][17:]
    hash_train.append(folder_path)

hash_val = []
for i in range(N_val):
    folder_path = split["alltest"][i][17:]
    hash_val.append(folder_path)

# Map data to train or val set
SUNRGBDMeta = loadmat("../SUNRGBDMeta3DBB_v2.mat")["SUNRGBDMeta"]

trainval_folder = "sunrgbd_trainval"
os.makedirs(trainval_folder, exist_ok=True)
f_train = os.path.join(trainval_folder, "train_data_idx.txt")
f_val = os.path.join(trainval_folder, "val_data_idx.txt")

with open(f_train, "w") as ft, open(f_val, "w") as fv:
    for imageId in range(len(SUNRGBDMeta)):
        data = SUNRGBDMeta[imageId]
        depthpath = data.depthpath[17:]
        # remove filename and containing folder from path 
        filepath = os.path.dirname(os.path.dirname(depthpath))
        if filepath in hash_train:
            ft.write(str(imageId) + "\n")
        elif filepath in hash_val:
            fv.write(str(imageId) + "\n")
