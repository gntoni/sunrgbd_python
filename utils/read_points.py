import numpy as np
from PIL import Image


def read_3d_pts_general(depthInpaint, K, depthInpaintsize, imageName=None, crop=(1, 1)):
    # Extract intrinsic camera parameters
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    # Identify invalid depth points
    invalid = depthInpaint == 0

    # Read the image if provided, otherwise create a default RGB image
    if imageName:
        im = Image.open(imageName)
        rgb = np.asarray(im).astype(np.float64) / 255.0  # Convert to double precision
    else:
        rgb = np.zeros((depthInpaintsize[0], depthInpaintsize[1], 3))
        rgb[:, :, 1] = 1  # Green channel set to 1

    rgb = rgb.reshape(-1, 3)

    # Generate 3D points
    x, y = np.meshgrid(np.arange(depthInpaintsize[1]), np.arange(depthInpaintsize[0]))
    x3 = (x - cx) * depthInpaint / fx
    y3 = (y - cy) * depthInpaint / fy
    z3 = depthInpaint

    points3dMatrix = np.stack((x3, z3, -y3), axis=-1)
    points3dMatrix[invalid] = np.nan

    points3d = np.stack((x3.ravel(), z3.ravel(), -y3.ravel()), axis=1)
    points3d[invalid.ravel()] = np.nan

    return rgb, points3d, points3dMatrix



def read3dPoints(data):
    # Read depth image
    depthVis = np.array(Image.open(data.depthpath))
    imsize = depthVis.shape

    # Process the depth image
    depthInpaint = np.bitwise_or(np.right_shift(depthVis, 3), np.left_shift(depthVis, 16 - 3))
    depthInpaint = depthInpaint.astype(np.float32) / 1000
    depthInpaint[depthInpaint > 8] = 8

    # Read 3D points and RGB image
    rgb, points3d, _ = read_3d_pts_general(depthInpaint, data.K, depthInpaint.shape, data.rgbpath)

    # Apply rotation
    points3d = (data.Rtilt @ points3d.T).T

    return rgb, points3d, depthInpaint, imsize

