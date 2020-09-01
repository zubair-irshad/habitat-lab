#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import random

import habitat_sim

import numpy as np

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

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


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
    config = habitat.get_config(
        config_paths="configs/test/habitat_r2r_vln_test.yaml"
    )
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()
    with SimpleRLEnv(config=config) as env:
        print("Environment creation successful")
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

        for episode in range(6):
            env.reset()            
            print(env.habitat_env.current_episode)
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
            time_step = 1.0 / (frame_skip * control_frequency)
            print("time_step = " + str(time_step))
            for action in control_sequence:
                # apply actions
                if continuous_nav:
                    # update the velocity control
                    # local forward is -z
                    vel_control.linear_velocity = np.array([0, 0, -action["forward_velocity"]])
                    # local up is y
                    vel_control.angular_velocity = np.array([0, action["rotation_velocity"], 0])

                    observations, reward, done, info = env.step(vel_control)
                    save_map(observations, info, images)
                    steps+=1

            print(f"Navigated to goal in {steps} steps.")
            images_to_video(images, dirname, str(episode_id), fps = int (1.0/time_step))
            images = []


if __name__ == "__main__":
    reference_path_example("geodesic_path")
