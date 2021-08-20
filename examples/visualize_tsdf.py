import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm

from vgn.grasp import Grasp, Label
from vgn.io import *
from vgn.perception import *
from vgn.simulation import ClutterRemovalSim
from vgn.utils.transform import Rotation, Transform

# o3d.visualization.webrtc_server.enable_webrtc()

OBJECT_COUNT_LAMBDA = 4
MAX_VIEWPOINT_COUNT = 6
RESOLUTION = 120

def main(args):
    sim = ClutterRemovalSim(args.scene, args.object_set, args.data_type, gui=args.sim_gui)

    object_count = np.random.poisson(OBJECT_COUNT_LAMBDA) + 1
    sim.reset(object_count)
    sim.save_state()

    # render synthetic depth images
    n = np.random.randint(MAX_VIEWPOINT_COUNT) + 1

    rgb_imgs, depth_imgs, extrinsics = render_images(sim, n)

    # reconstrct point cloud using a subset of the images
    # sim.size -> heuristically define grpper.finger depth * 6
    # volume RESOLUTION x RESOLUTION x RESOLUTION
    ctsdf = create_ctsdf(sim.size, RESOLUTION, rgb_imgs, depth_imgs, sim.camera.intrinsic, extrinsics)
    tsdf = create_tsdf(sim.size, RESOLUTION, depth_imgs, sim.camera.intrinsic, extrinsics)
    
    color_pcd = ctsdf.get_cloud()
    depth_pcd = tsdf.get_cloud()

    ctsdf.get_grid()
    tsdf.get_grid()

    # crop surface and borders from point cloud
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(sim.lower, sim.upper)
    color_pcd = color_pcd.crop(bounding_box)
    depth_pcd = depth_pcd.crop(bounding_box)

    o3d.visualization.draw_geometries([o3d.geometry.VoxelGrid.create_from_point_cloud(color_pcd, 100.0)])
    o3d.visualization.draw_geometries([o3d.geometry.VoxelGrid.create_from_point_cloud(depth_pcd, 100.0)])


def render_images(sim, n):
    height, width = sim.camera.intrinsic.height, sim.camera.intrinsic.width
    origin = Transform(Rotation.identity(), np.r_[sim.size / 2, sim.size / 2, 0.0])

    extrinsics = np.empty((n, 7), np.float32)
    depth_imgs = np.empty((n, height, width), np.float32)
    rgb_imgs =  np.empty((n, height, width, 3), np.int8)
    for i in range(n):
        r = np.random.uniform(1.6, 2.4) * sim.size
        theta = np.random.uniform(0.0, np.pi / 4.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        extrinsic = camera_on_sphere(origin, r, theta, phi)
        rgb_img, depth_img = sim.camera.render(extrinsic)

        extrinsics[i] = extrinsic.to_list()
        depth_imgs[i] = depth_img
        rgb_imgs[i] = rgb_img.astype(np.int8)

    return rgb_imgs, depth_imgs, extrinsics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="pile")
    parser.add_argument("--object-set", type=str, default="pile")
    parser.add_argument("--num-grasps", type=int, default=10000)
    parser.add_argument("--data-type", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--sim-gui", action="store_true")
    args = parser.parse_args()
    main(args)