import numpy as np

def hyperboloid_to_native_disk(t, x, y):
	return np.array([np.arccosh(t), np.arctan2(y, x)])


def euclidean_distance(coord):
    ang_diff = np.pi - np.abs(np.pi - np.abs(coord[:, 1]-coord[:, 3]))
    dist = coord[:, 0] * coord[:, 0] + coord[:, 2] * coord[:, 2]
    dist -= 2 * coord[:, 0] * coord[:, 2] * np.cos(ang_diff)
    return np.sqrt(dist).T


def euclidean_line(r_1, theta_1, r_2, theta_2):
	return [r_1, r_2], [theta_1, theta_2]


def native_disk_distance(coord):
    ang_diff = np.pi - np.abs(np.pi - np.abs(coord[:, 1] - coord[:, 3]))
    dist = np.cosh(coord[:, 0]) * np.cosh(coord[:, 2])
    dist -= np.sinh(coord[:, 0]) * np.sinh(coord[:, 2]) * np.cos(ang_diff)
    return np.arccosh(dist)


def native_disk_line(r_1, theta_1, r_2, theta_2, N=20):
    if r_1 < 1e-5 or r_2 < 1e-5 or abs((theta_1 - theta_2) % np.pi) < 1e-5:
    	return [r_1, r_2], [theta_1, theta_2]
    A_1, A_2 = np.tanh(r_1), np.tanh(r_2)
    theta_0 = np.arctan(- (A_1 * np.cos(theta_1) - A_2 * np.cos(theta_2)) / (A_1 * np.sin(theta_1) - A_2 * np.sin(theta_2)))
    B = A_1 * np.cos(theta_1 - theta_0)
    start_angle, end_angle = theta_1, theta_2
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle
    if np.pi < end_angle - start_angle:
        end_angle -= 2*np.pi
    theta = np.linspace(start_angle, end_angle, N)
    r = np.arctanh(B / np.cos(theta - theta_0))
    x, y = r*np.cos(theta), r*np.sin(theta)
    return np.sqrt(x*x + y*y), np.arctan2(y, x)