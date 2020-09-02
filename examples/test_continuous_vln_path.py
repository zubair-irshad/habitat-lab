#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import random
import quaternion
import habitat_sim
import gzip

import matplotlib.pyplot as plt
import json
import numpy as np
from habitat_sim.utils.common import quat_to_magnum
import magnum as mn

import habitat
from examples.shortest_path_follower_example import (
    SimpleRLEnv,
    draw_top_down_map,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import (
    append_text_to_image,
    images_to_video,
)

import compress_json

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


class ContinuousPathFollower(object):
    def __init__(self, sim, path, waypoint_threshold):
        self._sim = sim
        self._points = np.array(path[:])
        assert len(self._points) > 0
        self._length = self._sim.geodesic_distance(path[0], path[-1])
        self._threshold = waypoint_threshold
        self._step_size = 0.01
        self.progress = 0  # geodesic distance -> [0,1]
        self.waypoint = np.array(path[0])

        # setup progress waypoints
        _point_progress = [0]
        _segment_tangents = []
        _length = self._length
        for ix, point in enumerate(self._points):
            if ix > 0:
                segment = point - self._points[ix - 1]
                segment_length = np.linalg.norm(segment)
                segment_tangent = segment / segment_length
                _point_progress.append(
                    segment_length / _length + _point_progress[ix - 1]
                )
                # t-1 -> t
                _segment_tangents.append(segment_tangent)
        self._point_progress = _point_progress
        self._segment_tangents = _segment_tangents
        # final tangent is duplicated
        self._segment_tangents.append(self._segment_tangents[-1])

        # print("self._length = " + str(self._length))
        # print("num points = " + str(len(self._points)))
        # print("self._point_progress = " + str(self._point_progress))
        # print("self._segment_tangents = " + str(self._segment_tangents))

    def pos_at(self, progress):
        if progress <= 0:
            return self._points[0]
        elif progress >= 1.0:
            return self._points[-1]

        path_ix = 0
        for ix, prog in enumerate(self._point_progress):
            if prog > progress:
                path_ix = ix
                break

        segment_distance = self._length * (progress - self._point_progress[path_ix - 1])
        return (
            self._points[path_ix - 1]
            + self._segment_tangents[path_ix - 1] * segment_distance
        )

    def update_waypoint(self):
        if self.progress < 1.0:
            wp_disp = self.waypoint - self._sim.get_agent_state().position
            wp_dist = np.linalg.norm(wp_disp)
            node_pos = self._sim.get_agent_state().position
            step_size = self._step_size
            threshold = self._threshold
            while wp_dist < threshold:
                self.progress += step_size
                self.waypoint = self.pos_at(self.progress)
                if self.progress >= 1.0:
                    break
                wp_disp = self.waypoint - node_pos
                wp_dist = np.linalg.norm(wp_disp)


def setup_path_visualization(sim, path_follower, vis_samples=100):
    vis_ids = []
    sphere_handle = obj_attr_mgr.get_template_handles("uvSphereSolid")[0]
    sphere_template_cpy = obj_attr_mgr.get_template_by_handle(sphere_handle)
    sphere_template_cpy.scale *= 0.2
    template_id = obj_attr_mgr.register_template(sphere_template_cpy, "mini-sphere")
    print("template_id = " + str(template_id))
    if template_id < 0:
        return None
    vis_ids.append(sim.add_object_by_handle(sphere_handle))

    for point in path_follower._points:
        cp_id = sim.add_object_by_handle(sphere_handle)
        if cp_id < 0:
            print(cp_id)
            return None
        sim.set_translation(point, cp_id)
        vis_ids.append(cp_id)

    for i in range(vis_samples):
        cp_id = sim.add_object_by_handle("mini-sphere")
        if cp_id < 0:
            print(cp_id)
            return None
        sim.set_translation(path_follower.pos_at(float(i / vis_samples)), cp_id)
        vis_ids.append(cp_id)

    for id in vis_ids:
        if id < 0:
            print(id)
            return None

    for id in vis_ids:
        sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, id)

    return vis_ids

def remove_all_objects(sim):
    for id in sim.get_existing_object_ids():
        sim.remove_object(id)

def track_waypoint(waypoint, rs, vc, dt=1.0 / 60.0):
    angular_error_threshold = 0.5
    max_linear_speed = 1.0
    max_turn_speed = 1.0
    glob_forward = rs.rotation.transform_vector(mn.Vector3(0, 0, -1.0)).normalized()
    glob_right = rs.rotation.transform_vector(mn.Vector3(-1.0, 0, 0)).normalized()
    to_waypoint = mn.Vector3(waypoint) - rs.translation
    u_to_waypoint = to_waypoint.normalized()
    angle_error = float(mn.math.angle(glob_forward, u_to_waypoint))

    new_velocity = 0
    if angle_error < angular_error_threshold:
        # speed up to max
        new_velocity = (vc.linear_velocity[2] - max_linear_speed) / 2.0
    else:
        # slow down to 0
        new_velocity = (vc.linear_velocity[2]) / 2.0
    vc.linear_velocity = mn.Vector3(0, 0, new_velocity)

    # angular part
    rot_dir = 1.0
    if mn.math.dot(glob_right, u_to_waypoint) < 0:
        rot_dir = -1.0
    angular_correction = 0.0
    if angle_error > (max_turn_speed * 10.0 * dt):
        angular_correction = max_turn_speed
    else:
        angular_correction = angle_error / 2.0

    vc.angular_velocity = mn.Vector3(
        0, np.clip(rot_dir * angular_correction, -max_turn_speed, max_turn_speed), 0
    )
    return new_velocity, np.clip(rot_dir * angular_correction, -max_turn_speed, max_turn_speed)


def save_map(observations, info, images):
    im = observations["rgb"]
    top_down_map = draw_top_down_map(
        info, observations["heading"], im.shape[0]
    )
    output_im = np.concatenate((im, top_down_map), axis=1)
    output_im = append_text_to_image(
        output_im, observations["instruction"]["text"]
    )
    images.append(output_im)


def reference_path_example(mode):
    """
    Saves a video of a shortest path follower agent navigating from a start
    position to a goal. Agent follows the ground truth reference path by
    navigating to intermediate viewpoints en route to goal.
    Args:
        mode: 'geodesic_path' or 'greedy'
    """

    show_waypoint_indicators = False
    config = habitat.get_config(
        config_paths="configs/test/habitat_r2r_vln_test.yaml"
    )
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()

    split = 'train'
    vln_data_path = '/home/mirshad7/habitat-lab/data/datasets/vln/mp3d/r2r/v1/'+split+'/'+split+'.json.gz'
    with gzip.open(vln_data_path, "rt") as f:
        deserialized = json.loads(f.read())

    val_ids ={}
    for i in range(len(deserialized['episodes'])):
        val_ids[deserialized['episodes'][i]['episode_id']] = i 
    
    new_data_dict = {}
    new_data_dict['episodes']={}
    new_data_dict['instruction_vocab'] = deserialized['instruction_vocab']
    new_data_list = []
    save_fig=False
    steps_dict ={}

    with SimpleRLEnv(config=config) as env:
        print("Environment creation successful")

        # obj_attr_mgr = env.habitat_env._sim.get_object_template_manager()
        # remove_all_objects(env.habitat_env._sim)
        sim_time = 30  # @param {type:"integer"}
        continuous_nav = True  # @param {type:"boolean"}
        if continuous_nav:
            control_frequency = 10  # @param {type:"slider", min:1, max:30, step:1}
            frame_skip = 6  # @param {type:"slider", min:1, max:30, step:1}
        fps = control_frequency * frame_skip
        print("fps = " + str(fps))
        control_sequence = []
        for action in range(int(sim_time * control_frequency)):
            if continuous_nav:
                # allow forward velocity and y rotation to vary
                control_sequence.append(
                    {
                        "forward_velocity": random.random() * 2.0,  # [0,2)
                        "rotation_velocity": (random.random() - 0.5) * 2.0,  # [-1,1)
                    }
                )
            else:
                control_sequence.append(random.choice(action_names))

                # create and configure a new VelocityControl structure
        vel_control = habitat_sim.physics.VelocityControl()
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = True
        vel_control.ang_vel_is_local = True

        vis_ids = []

        collided_trajectories = []
        trajectory_without_collision = True

        for episode in range(len(deserialized['episodes'])):
            counter=0
            env.reset()            
            episode_id = env.habitat_env.current_episode.episode_id
            print(
                f"Agent stepping around inside environment. Episode id: {episode_id}"
            )

            dirname = os.path.join(
                IMAGE_DIR, "vln_reference_path_example", mode, "%02d" % episode
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)

            images = []
            steps = 0
            reference_path = env.habitat_env.current_episode.reference_path + [
                env.habitat_env.current_episode.goals[0].position
            ]

            # manually control the object's kinematic state via velocity integration
            time_step = 1.0 / (30)

            x=[]
            y=[]
            yaw=[]
            vel=[]
            omega=[]

            continuous_path_follower = ContinuousPathFollower(
            env.habitat_env._sim, reference_path, waypoint_threshold=0.4)
            max_time = 30.0
            done = False
            EPS = 1e-4
            prev_pos = np.linalg.norm(env.habitat_env._sim.get_agent_state().position)
            if show_waypoint_indicators:
                for id in vis_ids:
                    sim.remove_object(id)
                vis_ids = setup_path_visualization(env.habitat_env._sim, continuous_path_follower)
            
            while continuous_path_follower.progress < 1.0:
                # print("done",done)
                if done:
                    break
                if counter == 150:
                    counter = 0
                    collided_trajectories.append(env.habitat_env.current_episode.episode_id)
                    trajectory_without_collision = False
                    break
                continuous_path_follower.update_waypoint()

                if show_waypoint_indicators:
                    sim.set_translation(continuous_path_follower.waypoint, vis_ids[0])
                
                agent_state = env.habitat_env._sim.get_agent_state()
                pos = np.linalg.norm(env.habitat_env._sim.get_agent_state().position)

                if abs(pos - prev_pos)<EPS:
                    counter+=1 
                previous_rigid_state = habitat_sim.RigidState(
                    quat_to_magnum(agent_state.rotation), agent_state.position
                )

                v,w = track_waypoint(
                    continuous_path_follower.waypoint,
                    previous_rigid_state,
                    vel_control,
                    dt=time_step,
                )

                observations, reward, done, info = env.step(vel_control)
                # print(observations)
                # save_map(observations, info, images)
                prev_pos = pos
                x.append(env.habitat_env._sim.get_agent_state().position[0])
                y.append(-env.habitat_env._sim.get_agent_state().position[2])
                yaw.append(quaternion.as_euler_angles(env.habitat_env._sim.get_agent_state().rotation)[1])
                vel.append(v)
                omega.append(w)
                steps+=1

            if save_fig:  # pragma: no cover
                plt.close()
                plt.subplots(1)
                plt.plot(x, y, "xb", label="input")
                plt.grid(True)
                plt.axis("equal")
                plt.xlabel("x[m]")
                plt.ylabel("y[m]")
                plt.legend()
                pose_title = dirname+'pose.png'
                plt.savefig(pose_title)

                plt.subplots(1)
                plt.plot(yaw, "-r", label="yaw")
                plt.grid(True)
                plt.legend()
                plt.xlabel("line length[m]")
                plt.ylabel("yaw angle[deg]")
                yaw_title = dirname+'yaw.png'
                plt.savefig(yaw_title)

                plt.subplots(1)
                plt.plot(vel, "-r", label="vel")
                plt.grid(True)
                plt.legend()
                plt.xlabel("line length[m]")
                plt.ylabel("omega_reference [rad/s^2]")
                vel_title = dirname+'vel.png'
                plt.savefig(vel_title)

                plt.subplots(1)
                plt.plot(omega, "-r", label="v_reference")
                plt.grid(True)
                plt.legend()
                plt.xlabel("line length[m]")
                plt.ylabel("v_reference [m/s]")
                omega_title = dirname+'omega.png'
                plt.savefig(omega_title)

                x=[]
                y=[]
                yaw=[]
                vel=[]
                omega=[]

            if trajectory_without_collision:
                ids = val_ids[episode_id]
                single_data_dict = deserialized['episodes'][ids]
                new_data_list.append(single_data_dict)
            trajectory_without_collision = True
            print(f"Navigated to goal in {steps} steps.")
            steps_dict[episode] = steps
            # images_to_video(images, dirname, str(episode_id), fps = int (1.0/time_step))
            images = []


        steps_path =  '/home/mirshad7/habitat-lab/train_steps.json.gz'       
        # new_data_dict['episodes'] = new_data_list 
        # path = '/home/mirshad7/habitat-lab/data/datasets/vln/mp3d/r2r/robo_vln/train/train.json.gz'
        # compress_json.dump(new_data_dict, path)
        compress_json.dump(steps_dict, steps_path)
        print("collided trajectories:", collided_trajectories)


if __name__ == "__main__":
    reference_path_example("geodesic_path")
