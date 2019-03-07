import load_data as ld
import numpy as np
from SLAM_helper import *
import matplotlib.pyplot as plt
import MapUtils as maput
import cv2
import random


####Dataset####

joint = ld.get_joint("data/train_joint2")
lid = ld.get_lidar("data/train_lidar2")

itv = 1 # of skip in drawing maps

###############

angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.0])

N, N_threshold = 100, 35
particles = np.zeros((N, 3))
weight = np.einsum('..., ...', 1.0 / N, np.ones((N, 1)))

mapfig = {}
mapfig['res'] = 0.05
mapfig['xmin'] = -40
mapfig['ymin'] = -40
mapfig['xmax'] = 40
mapfig['ymax'] = 40
mapfig['sizex'] = int(np.ceil((mapfig['xmax'] - mapfig['xmin']) / mapfig['res'] + 1))
mapfig['sizey'] = int(np.ceil((mapfig['ymax'] - mapfig['ymin']) / mapfig['res'] + 1))

mapfig['log_map'] = np.zeros((mapfig['sizex'], mapfig['sizey']))
mapfig['map'] = np.zeros((mapfig['sizex'], mapfig['sizey']), dtype = np.int8)
mapfig['show_map'] = 0.5 * np.ones((mapfig['sizex'], mapfig['sizey'], 3), dtype = np.int8)

pos_phy, posX_map, posY_map = {}, {}, {}

factor = np.array([1, 1, 10])

x_im = np.arange(mapfig['xmin'], mapfig['xmax'] + mapfig['res'], mapfig['res'])  # x-positions of each pixel of the map
y_im = np.arange(mapfig['ymin'], mapfig['ymax'] + mapfig['res'], mapfig['res'])  # y-positions of each pixel of the map

x_range = np.arange(-0.05, 0.06, 0.05)
y_range = np.arange(-0.05, 0.06, 0.05)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_title("SLAM Map")

im = ax.imshow(mapfig['show_map'], cmap = "hot")
fig.show()

ts = joint['ts']
h_angle = joint['head_angles']
rpy_robot = joint['rpy']

lid_p = lid[0]
rpy_p = lid_p['rpy']
ind_0 = np.argmin(np.absolute(ts - lid_p['t'][0][0]))
pos_phy, posX_map, posY_map = mapConvert(lid_p['scan'], rpy_robot[:, ind_0], h_angle[:, ind_0], angles, particles, N, pos_phy, posX_map, posY_map, mapfig)
mapfig = drawMap(particles[0, :], posX_map[0], posY_map[0], mapfig)

pose_p, yaw_p = lid_p['pose'], rpy_p[0, 2]
timeline = len(lid)
for i in xrange(1, timeline):
	print "{0}/{1}".format(i,timeline)
	lid_c = lid[i]
	pose_c, rpy_c = lid_c['pose'], lid_c['rpy']
	yaw_c = rpy_c[0, 2]

	yaw_est = particles[:, 2]

	delta_x_gb = pose_c[0][0] - pose_p[0][0]
	delta_y_gb = pose_c[0][1] - pose_p[0][1]
	delta_theta_gb = yaw_c - yaw_p

	delta_x_lc = np.einsum('..., ...', np.cos(yaw_p), delta_x_gb) + np.einsum('..., ...', np.sin(yaw_p), delta_y_gb)
	delta_y_lc = np.einsum('..., ...', -np.sin(yaw_p), delta_x_gb) + np.einsum('..., ...', np.cos(yaw_p), delta_y_gb)
	delta_theta_lc = delta_theta_gb

	delta_x_gb_new = (np.einsum('..., ...', np.cos(yaw_est), delta_x_lc) - np.einsum('..., ...', np.sin(yaw_est), delta_y_lc)).reshape(-1, N)
	delta_y_gb_new = (np.einsum('..., ...', np.sin(yaw_est), delta_x_lc) + np.einsum('..., ...', np.cos(yaw_est), delta_y_lc)).reshape(-1, N)
	delta_theta_gb_new = np.tile(delta_theta_lc, (1, N))

	ut = np.concatenate([np.concatenate([delta_x_gb_new, delta_y_gb_new], axis=0), delta_theta_gb_new], axis=0)
	ut = np.einsum('ji', ut)

	noise = np.einsum('..., ...', factor, np.random.normal(0, 1e-3, (N, 1)))
	particles = particles + ut + noise

	scan_c = lid_c['scan']
	ind_i = np.argmin(np.absolute(ts - lid_c['t'][0][0]))
	pos_phy, posX_map, posY_map = mapConvert(scan_c, rpy_robot[:, ind_i], h_angle[:, ind_i], angles, particles, N, pos_phy, posX_map, posY_map, mapfig)

	corr = np.zeros((N, 1))

	for i in range(N):
		size = pos_phy[i].shape[1]
		Y = np.concatenate([pos_phy[i], np.zeros((1, size))], axis = 0)
		corr_cur = maput.mapCorrelation(mapfig['map'], x_im, y_im, Y[0 : 3, :], x_range, y_range)
		ind = np.argmax(corr_cur)

		corr[i] = corr_cur[ind / 3, ind % 3]
		particles[i, 0] += x_range[ind / 3]
		particles[i, 1] += y_range[ind % 3]

	wtmp = np.log(weight) + corr
	wtmp_max = wtmp[np.argmax(wtmp)]
	lse = np.log(np.sum(np.exp(wtmp - wtmp_max)))
	wtmp = wtmp - wtmp_max - lse

	weight = np.exp(wtmp)
	ind_best = weight.argmax()


	x_r = (np.ceil((particles[ind_best, 0] - mapfig['xmin']) / mapfig['res']).astype(np.int16) - 1)
	y_r = (np.ceil((particles[ind_best, 1] - mapfig['xmin']) / mapfig['res']).astype(np.int16) - 1)
	mapfig['show_map'][x_r, y_r, 0] = 255

	mapfig = drawMap(particles[ind_best, :], posX_map[ind_best], posY_map[ind_best], mapfig)

	pose_p, yaw_p = pose_c, yaw_c

	# Resampling
	N_eff = 1 / np.sum(np.square(weight))
	if N_eff < N_threshold:
		#particles = resample(N, weight, particles)

		particle_New = np.zeros((N, 3))
		r = random.uniform(0, 1.0 / N)

		c, i = weight[0], 0
		for m in range(N):
			u = r + m * (1.0 / N)
			
			while u > c:
				i = i + 1
				c = c + weight[i]

		particle_New[m, :] = particles[i, :]
		particles = particle_New
		
		weight = np.einsum('..., ...', 1.0 / N, np.ones((N, 1)))

	#ax.imshow(mapfig['show_map'], cmap = "hot")
	#im.set_data(mapfig['show_map'])
	#im.axes.figure.canvas.draw()

fig1 = plt.figure(1)
plt.imshow(mapfig['show_map'], cmap = "hot")
plt.show()
