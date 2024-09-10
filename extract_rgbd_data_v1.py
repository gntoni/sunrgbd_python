import os
import numpy as np
from utils.load_mat import loadmat

SUNRGBDMeta = loadmat("../SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat")["SUNRGBDMeta"]

det_label_folder = 'sunrgbd_trainval/label_v1/'
os.makedirs(det_label_folder, exist_ok=True)

for imageId in range(len(SUNRGBDMeta)):
    data = SUNRGBDMeta[imageId]
    data.depthpath = data.depthpath[17:]
    data.rgbpath = data.depthpath[17:]

    txt_filename = f"{imageId:06d}.txt"

    if not isinstance(data.groundtruth3DBB, np.ndarray):
        data.groundtruth3DBB = [data.groundtruth3DBB]
    # Write 2D and 3D box label 
    with open(os.path.join(det_label_folder, txt_filename), "w") as fid:
        for j in range(len(data.groundtruth3DBB)):
            centroid = data.groundtruth3DBB[j].centroid
            classname = data.groundtruth3DBB[j].classname
            orientation = data.groundtruth3DBB[j].orientation
            coeffs = abs(data.groundtruth3DBB[j].coeffs)
            if len(data.groundtruth3DBB[j].gtBb2D) == 4:
                box2d = data.groundtruth3DBB[j].gtBb2D
            else:
                box2d = [0,0,0,0]

            fid.write('{} {} {} {} {} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                classname, 
                int(box2d[0]), int(box2d[1]), int(box2d[2]), int(box2d[3]),
                centroid[0], centroid[1], centroid[2], 
                coeffs[0], coeffs[1], coeffs[2], 
                orientation[0], orientation[1]
            ))

