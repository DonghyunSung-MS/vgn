from pathlib2 import Path
import time

import numpy as np
import pybullet
import scipy.stats as stats

from vgn import Label
from vgn.perception import *
from vgn.utils import btsim, io, vis
from vgn.utils.transform import Rotation, Transform


class GraspSimulation(object):
    def __init__(self, object_set, config_path, gui=True):
        assert object_set in ["blocks", "train", "test", "adversarial"]
        self._config = io.load_dict(Path(config_path))

        self._urdf_root = Path(self._config["urdf_root"])
        self._object_set = object_set
        self._discover_object_urdfs()
        self._test = False if object_set == "train" else True
        self._gui = gui

        self.size = 4 * self._config["max_opening_width"]
        self.world = btsim.BtWorld(self._gui)

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self._setup_table()
        self._setup_camera()
        self._draw_task_space()
        self._generate_heap(object_count)

    def acquire_tsdf(self, num_viewpoints=6):
        tsdf = TSDFVolume(self.size, 40)
        high_res_tsdf = TSDFVolume(self.size, 120)

        t_world_center = np.r_[0.5 * self.size, 0.5 * self.size, 0.0]
        T_world_center = Transform(Rotation.identity(), t_world_center)

        for i in range(num_viewpoints):
            phi = 2.0 * np.pi * i / num_viewpoints
            theta = np.pi / 4.0
            r = 1.5 * self.size
            extrinsic = compute_viewpoint_on_hemisphere(T_world_center, phi, theta, r)
            depth_img = self.camera.render(extrinsic)[1]
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
        pc = high_res_tsdf.extract_point_cloud()

        return tsdf, pc

    def execute_grasp(self, grasp, remove=False):
        T_world_grasp = grasp.pose

        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp
        T_world_retreat = T_world_grasp * T_grasp_retreat

        gripper = Gripper(self.world, self._config)
        gripper.set_tcp(T_world_pregrasp)

        if gripper.detect_collision(threshold=0.0):
            result = Label.FAILURE, 0.0
        else:
            gripper.move_tcp_xyz(T_world_grasp)
            if gripper.detect_collision():
                result = Label.FAILURE, 0.00
            else:
                gripper.move(0.0)
                gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self._check_success(gripper):
                    result = Label.SUCCESS, gripper.read()
                    if remove:
                        contacts = self.world.check_contacts(gripper._body)
                        self.world.remove_body(contacts[0].bodyB)
                else:
                    result = Label.FAILURE, 0.0
        del gripper

        return result

    def _discover_object_urdfs(self):
        root = self._urdf_root / self._object_set
        self._urdfs = [d / (d.name + ".urdf") for d in root.iterdir() if d.is_dir()]

    def _setup_table(self):
        plane = self.world.load_urdf(
            self._urdf_root / "plane/plane.urdf",
            Transform(Rotation.identity(), [0.0, 0.0, 0.0]),
        )

    def _setup_camera(self):
        intrinsic = PinholeCameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

    def _generate_heap(self, object_count):
        urdfs = np.random.choice(self._urdfs, size=object_count)
        for urdf in urdfs:
            planar_position = self._sample_planar_position()
            pose = Transform(Rotation.random(), np.r_[planar_position, 0.15])
            scale = 1.0 if self._test else np.random.uniform(0.8, 1.0)
            self._drop_object(urdf, pose, scale)

    def _sample_planar_position(self):
        l, u = 0.0, self.size
        mu, sigma = self.size / 2.0, self.size / 4.0
        X = stats.truncnorm((l - mu) / sigma, (u - mu) / sigma, loc=mu, scale=sigma)
        return X.rvs(2)

    def _drop_object(self, model_path, pose, scale=1.0):
        body = self.world.load_urdf(model_path, pose, scale=scale)
        for _ in range(240):
            self.world.step()

    def _check_success(self, gripper):
        # TODO this can be improved
        return gripper.read() > 0.1 * gripper.max_opening_width

    def _draw_task_space(self):
        points = vis.workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )


class Gripper(object):
    def __init__(self, world, config):
        self._world = world
        self._urdf_path = Path(config["urdf_root"]) / "panda/hand.urdf"
        self._body = None
        self._T_tool0_tcp = Transform.from_dict(config["T_tool0_tcp"])
        self._T_tcp_tool0 = self._T_tool0_tcp.inverse()

        self.max_opening_width = config["max_opening_width"]

    def __del__(self):
        self._world.remove_body(self._body)

    def set_tcp(self, target):
        T_world_tool0 = target * self._T_tcp_tool0

        if self._body is None:  # spawn robot if necessary
            self._body = self._world.load_urdf(self._urdf_path, T_world_tool0)
            self._constraint = self._world.add_constraint(
                self._body,
                None,
                None,
                None,
                pybullet.JOINT_FIXED,
                [0.0, 0.0, 0.0],
                Transform.identity(),
                T_world_tool0,
            )
            self._finger_l = self._body.joints["panda_finger_joint1"]
            self._finger_l.set_position(0.5 * self.max_opening_width, kinematics=True)
            self._finger_r = self._body.joints["panda_finger_joint2"]
            self._finger_r.set_position(0.5 * self.max_opening_width, kinematics=True)

        self._body.set_pose(T_world_tool0)
        self._constraint.change(T_world_tool0, max_force=300)
        self._world.step()

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        pose = self._body.get_pose() * self._T_tool0_tcp

        pos_diff = target.translation - pose.translation
        n_steps = int(np.linalg.norm(pos_diff) / eef_step)
        dist_step = pos_diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            pose.translation += dist_step
            self._constraint.change(pose * self._T_tcp_tool0, max_force=300)
            for _ in range(int(dur_step / self._world.dt)):
                self._world.step()
            if abort_on_contact and self.detect_collision():
                return

    def detect_collision(self, threshold=10):
        contacts = self._world.check_contacts(self._body)
        for contact in contacts:
            if contact.force > threshold:
                return True
        return False

    def move(self, width):
        self._finger_l.set_position(0.5 * width)
        self._finger_r.set_position(0.5 * width)
        for _ in range(int(0.5 / self._world.dt)):
            self._world.step()

    def read(self):
        width = self._finger_l.get_position() + self._finger_r.get_position()
        return width
