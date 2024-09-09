import os
from utils.load_mat import loadmat

SUNRGBDMeta = loadmat("SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat")["SUNRGBDMeta"]

# Create folders
det_label_folder = '../sunrgbd_trainval/label_v1/'
os.makedirs(det_label_folder, exist_ok=False)

# Read

for imageId in range(0, 10334):
    data = SUNRGBDMeta[imageId]
    data.depthpath = data.depthpath[17:]
    data.rgbpath = data.depthpath[17:]

    txt_filename = f"{imageId:06d}.txt"
    
    # Write 2D and 3D box label 
    with open(os.path.join(det_label_folder, txt_filename), "w"):
        for j in range(len(data.groundtruth3DBB)):
            centroid = data.groundtruth3DBB[j].centroid
            classname = data.groundtruth3DBB[j].classname
            orientation = data.groundtruth3DBB[j].orientation
            coeffs = abs(data.groundtruth3DBB[j].coeffs)
            box2d = data.groundtruth3DBB[j].gtBb2D

