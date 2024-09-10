import os
import shutil
import scipy
import numpy as np
from multiprocessing import Pool
from utils.load_mat import loadmat
from utils.read_points import read3dPoints


# Add here the location of the folder containing the original dataset
# .mat files (SUNRGBDMeta3DBB_v2.mat and SUNRGBDMeta2DBB_v2.mat)
sunrgbd_folder = ".."

SUNRGBDMeta = loadmat(
    os.path.join(sunrgbd_folder, "SUNRGBDMeta3DBB_v2.mat"))["SUNRGBDMeta"]
SUNRGBDMeta2DBB = loadmat(
    os.path.join(sunrgbd_folder, "SUNRGBDMeta2DBB_v2.mat"))["SUNRGBDMeta2DBB"]

depth_folder = "sunrgbd_trainval/depth/"
image_folder = "sunrgbd_trainval/image/"
calib_folder = "sunrgbd_trainval/calib/"
det_label_folder = "sunrgbd_trainval/label/"
seg_label_folder = "sunrgbd_trainval/seg_label"
os.makedirs(depth_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)
os.makedirs(calib_folder, exist_ok=True)
os.makedirs(det_label_folder, exist_ok=True)
os.makedirs(seg_label_folder, exist_ok=True)

# Processing function for each image
def process_image(imageId):
    try:
        data = SUNRGBDMeta[imageId]
        data.depthpath = os.path.join(sunrgbd_folder, data.depthpath[17:])
        data.rgbpath = os.path.join(sunrgbd_folder, data.rgbpath[17:])

        # Write point cloud in depth map
        rgb, points3d, depthInpaint, imsize = read3dPoints(data)
        valid_mask = ~np.isnan(points3d[:, 0])
        rgb = rgb[valid_mask]
        points3d = points3d[valid_mask]
        points3d_rgb = np.hstack([points3d, rgb])

        mat_filename = f"{imageId:06d}.mat"
        scipy.io.savemat(os.path.join(depth_folder, mat_filename), {'points3d_rgb': points3d_rgb})

        # Write images
        shutil.copy(data.rgbpath, os.path.join(image_folder, f"{imageId:06d}.jpg"))

        # Write calibration
        txt_filename = f"{imageId:06d}.txt"
        np.savetxt(os.path.join(calib_folder, txt_filename), data.Rtilt.flatten(), delimiter=' ')
        with open(os.path.join(calib_folder, txt_filename), 'a') as f:
            np.savetxt(f, data.K.flatten(), delimiter=' ')

        # Write 2D and 3D box labels
        data2d = SUNRGBDMeta2DBB[imageId]
        if not isinstance(data.groundtruth3DBB, np.ndarray):
            data.groundtruth3DBB = [data.groundtruth3DBB]
        if not isinstance(data2d.groundtruth2DBB, np.ndarray):
            data2d.groundtruth2DBB = [data2d.groundtruth2DBB]
        with open(os.path.join(det_label_folder, txt_filename), 'w') as f:
            for j in range(len(data.groundtruth3DBB)):
                box2d = data2d.groundtruth2DBB[j].gtBb2D
                centroid = data.groundtruth3DBB[j].centroid
                classname = data.groundtruth3DBB[j].classname
                orientation = data.groundtruth3DBB[j].orientation
                coeffs = np.abs(data.groundtruth3DBB[j].coeffs)
                assert data2d.groundtruth2DBB[j].classname == classname
                f.write(f"{classname} {box2d[0]} {box2d[1]} {box2d[2]} {box2d[3]} {centroid[0]} {centroid[1]} "
                        f"{centroid[2]} {coeffs[0]} {coeffs[1]} {coeffs[2]} {orientation[0]} {orientation[1]}\n")

    except Exception as e: 
        print(f"Error processing image {imageId}: {e}")

# Parallel processing
if __name__ == '__main__':
    with Pool() as p:
        p.map(process_image, range(len(SUNRGBDMeta)))
