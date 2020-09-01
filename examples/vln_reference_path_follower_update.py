#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import gzip
import json
import io

import gzip, pickle

from bson import json_util
import compress_json

import numpy as np
import quaternion

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
    # config = habitat.get_config(
    #     config_paths="configs/test/habitat_r2r_vln_test_continuous.yaml"
    # )
    config = habitat.get_config(
        config_paths="configs/test/habitat_r2r_vln_test.yaml"
    )
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.freeze()

    vln_data_path = '/home/mirshad7/habitat-lab/data/datasets/vln/mp3d/r2r/v1/val_seen/val_seen.json.gz'
    with gzip.open(vln_data_path, "rt") as f:
        deserialized = json.loads(f.read())

    new_data_dict = {}
    new_data_dict['episodes']={}
    new_data_dict['instruction_vocab'] = deserialized['instruction_vocab']

    new_data_list = []

    with SimpleRLEnv(config=config) as env:
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius=0.5, return_one_hot=False
        )
        follower.mode = mode
        print("Environment creation successful")

        for episode in range(50):
            positions = []
            rotations = []
            env.reset()

            episode_id = env.habitat_env.current_episode.episode_id
            new_data_dict['episodes'][episode_id-1] ={}

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

            positions.append(env.habitat_env._sim.get_agent_state().position.tolist())
            for point in reference_path:
                done = False
                while not done:
                    best_action = follower.get_next_action(point)
                    if best_action == None or best_action ==0:
                        break
                    observations, reward, done, info = env.step(best_action)

                    pos =  env.habitat_env._sim.get_agent_state().position.tolist()
                    positions.append(pos)
                    save_map(observations, info, images)
                    steps += 1
            single_data_dict = deserialized['episodes'][episode_id-1]
            single_data_dict['reference_path'] = positions
            new_data_list.append(single_data_dict)
            print(f"Navigated to goal in {steps} steps.")
            images_to_video(images, dirname, str(episode_id))
            images = []

        # dump = json.dumps(new_data_dict,  default=json_util.default)
        # with io.open('new_dataset.json.gz', 'w') as outfile:
        #     dump = json.dumps(new_data_dict,  default=json_util.default)
        #     outfile.write(dump)
        # serialized_obj = pickle.dumps(new_data_dict)
        # with gzip.GzipFile('new_dataset.json.gz', 'wb') as fout:   # 4. gzip
        #     fout.write(serialized_obj) 
        new_data_dict['episodes'] = new_data_list 
        path = '/home/mirshad7/habitat-lab/data/datasets/vln/mp3d/r2r/v1/test/test.json.gz'
        compress_json.dump(new_data_dict, path)


if __name__ == "__main__":
    reference_path_example("geodesic_path")
