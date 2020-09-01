#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import random
import time
import habitat_sim

from habitat_sim.utils.common import quat_to_magnum

import numpy as np
import quaternion
import magnum as mn

import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import json

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
try:
	import cubic_spline_planner
except ImportError:
	raise

lqr_Q =  np.eye(3)
# lqr_Q[0,0] = 0.001
# lqr_Q[1,1] = 0.001
lqr_Q[2,2] = 50

lqr_R = np.eye(2)
# lqr_R[1,1] = 0.1

# kpv = 0.005
kpw = 0.05


L   =  0.1  # Wheel base of the vehicle [m]
rad =  0.035
show_animation = True
process_done = False
start_time = time.time()

class State:

	def __init__(self, x=0.0, y=0.0, yaw=0.0):
		self.x = x
		self.y = y
		self.yaw = yaw

def update(env,state,a, delta, vel_control, time_step):
	dt = time_step
	# angle = quaternion.as_euler_angles(env.habitat_env._sim.get_agent_state().rotation)

	agent_forward = quat_to_magnum(env.habitat_env._sim.get_agent_state().rotation
        ).transform_vector(mn.Vector3(0, 0, -1.0))

	# print("linear velocity", state.v)
	# print("angular velocity", state.yaw)
	state.x = env.habitat_env._sim.get_agent_state().position[0]
	state.y = -env.habitat_env._sim.get_agent_state().position[2]
	# state.yaw = -angle[1]
	state.yaw = math.atan2(agent_forward[0], agent_forward[2])

	vel_control.linear_velocity = np.array([0, 0, -a])
	# local up is y
	vel_control.angular_velocity = np.array([0, delta, 0])

	print("lin", a)
	print("vel", delta)
	
	obs, reward, done, info = env.step(vel_control)
	return obs, info, state

def solve_dare(A, B, Q, R):
	"""
	solve a discrete time_Algebraic Riccati equation (DARE)
	"""
	x = Q
	x_next = Q
	max_iter = 150
	eps = 0.0001
	for i in range(max_iter):
		x_next = np.dot(np.dot(A.T, x), A) - np.dot(np.dot(np.dot(np.dot(A.T,x), B), la.inv(R + np.dot(np.dot(B.T, x), B))), np.dot(np.dot(B.T, x), A)) + Q
		if (abs(x_next - x)).max() < eps:
			break
		x = x_next

	return x_next

def dlqr(A, B, Q, R):
	"""Solve the discrete time lqr controller.
	x[k+1] = A x[k] + B u[k]
	cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
	# ref Bertsekas, p.151
	"""

	# first, try to solve the ricatti equation
	X = solve_dare(A, B, Q, R)

	K = np.dot(la.inv(R + np.dot(np.dot(B.T,X),B)), np.dot(np.dot(B.T,X),A))
	return K, X

def lqr_diff_drive_control(state, cx, cy, cyaw, ck, s_profile, pe, pth_e, time_step):
	dt = time_step
	ind, dx,dy = calc_nearest_index(state, cx, cy, cyaw)

	if abs(ind-len(cx))<=2:
		ind = ind-2

	w_d = ck[ind+5]
	# th_e = pi_2_pi(cyaw[ind+1] - state.yaw)
	th_e = cyaw[ind+2] - state.yaw
	v_d     =  s_profile[ind]
	
	A = np.array([    [0, w_d, 0],
					  [-w_d, 0,  v_d],
					  [0, 0, 0]])

	B = np.array([[1, 0],
				  [0, 0],
				  [0, 1]])

	
	K, _ = dlqr(A, B, lqr_Q, lqr_R)

	x = np.zeros((3, 1))

	x[0, 0] = dx
	x[1, 0] = dy
	x[2, 0] = th_e
	
	u1= np.dot(-K, x)[0, 0]
	u2= np.dot(-K, x)[1, 0]
	return u1, u2, v_d, w_d, th_e, ind+2


def pi_2_pi(angle):
	return (angle + math.pi) % (2 * math.pi) - math.pi


def calc_nearest_index(state, cx, cy, cyaw):
	dx = [state.x - icx for icx in cx]
	dy = [state.y - icy for icy in cy]

	d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

	mind = min(d)

	ind = d.index(mind)

	mind = math.sqrt(mind)

	dxl = cx[ind] - state.x
	dyl = cy[ind] - state.y

	# angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
	# if angle < 0:
	# 	mind *= -1

	return ind, dxl, dyl

def closed_loop_prediction(env,state,cx, cy, cyaw, ck, s_profile, goal,images, vel_control, time_step):

	image_track = 0
	T = 100.0  # max simulation time
	goal_dis  = 0.5

	stop_dist = 0.2
	
	vel=[0.0]
	omeg=[0.0]

	time = 0.0
	x = [state.x]
	y = [state.y]
	yaw = [state.yaw]
	t = [0.0]

	e, e_th = 0.0, 0.0

	pose        = []
	action      = []

	pose_list   = []
	action_list = []
	images=[]
	sem = []
	output_im =[] 

	while T >= time:
		u1, u2, v_d, w_d, the_e, target_ind = lqr_diff_drive_control(
			state, cx, cy, cyaw, ck, s_profile, e, e_th, time_step)

		the_e = np.arctan2(np.sin(the_e),np.cos(the_e))
		v_input = v_d*np.cos(the_e) - u1
		w_input = w_d - u2
		obs, info,state = update(env, state, v_input, w_input, vel_control, time_step)
		save_map(obs, info, images)

		time = time + time_step

		# check goal
		dx = state.x - goal[0]
		dy = state.y - goal[1]

		if math.hypot(dx, dy) <= goal_dis:
			print("Goal")
			break

		x.append(state.x)
		y.append(state.y)
		yaw.append(state.yaw)
		vel.append(v_input)
		omeg.append(w_input)
		# v.append(state.v)
		t.append(time)

		plt.cla()
		#   # for stopping simulation with the esc key.
		plt.gcf().canvas.mpl_connect('key_release_event',
				lambda event: [exit(0) if event.key == 'escape' else None])
		plt.plot(cx, cy, "-r", label="course")
		plt.plot(x, y, "ob", label="trajectory")
		plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
		plt.axis("equal")
		plt.grid(True)
		plt.pause(0.0001)

		image_track+=1

	return t, x, y, yaw, vel, omeg,images

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
		config_paths="configs/test/habitat_r2r_vln_test_continuous.yaml"
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

			reference_path = env.habitat_env.current_episode.reference_path

			waypoints = np.array(reference_path)

			for i in range(waypoints.shape[1]-1):
				if np.abs(np.linalg.norm(waypoints[i+1,:]- waypoints[i,:]))<5:
					waypoints = np.delete(waypoints, (i+1), axis=0)

			waypoints = waypoints[0::10]
			x_wp = waypoints[:,0]
			y_wp = -waypoints[:,2]
			z_wp = waypoints[:,1]
			
			ax = x_wp
			ay = y_wp

			cx, cy, cyaw, ck, s_profile, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.01)

			goal = [cx[-1], cy[-1], cyaw[-1]]

			agent_forward = quat_to_magnum(env.habitat_env._sim.get_agent_state().rotation
			).transform_vector(mn.Vector3(0, 0, -1.0))

			angle = quaternion.as_euler_angles(env.habitat_env._sim.get_agent_state().rotation)
			init_x = env.habitat_env._sim.get_agent_state().position[0]
			init_y = -env.habitat_env._sim.get_agent_state().position[2]
			# init_yaw = angle[1]

			init_yaw = math.atan2(agent_forward[0], agent_forward[2])

			state = State(x=init_x, y=init_y, yaw=init_yaw)

			time_step = 1.0 / (30)
			T=500
			t, x, y, yaw, vel,omeg,images = closed_loop_prediction(env,state,cx, cy, cyaw, ck, s_profile, goal, images, vel_control, time_step)
			images_to_video(images, dirname, str(episode_id), fps = int (1.0/time_step))
			images = []
			# # manually control the object's kinematic state via velocity integration
			
			# print("time_step = " + str(time_step))
			# for action in control_sequence:
			#     # apply actions
			#     if continuous_nav:
			#         # update the velocity control
			#         # local forward is -z
			#         vel_control.linear_velocity = np.array([0, 0, -action["forward_velocity"]])
			#         # local up is y
			#         vel_control.angular_velocity = np.array([0, action["rotation_velocity"], 0])

			#         observations, reward, done, info = env.step(vel_control)
			#         save_map(observations, info, images)
			#         steps+=1

			# print(f"Navigated to goal in {steps} steps.")
			# images_to_video(images, dirname, str(episode_id), fps = int (1.0/time_step))
			# images = []


if __name__ == "__main__":
	reference_path_example("geodesic_path")
