import numpy as np
from copy import deepcopy


def hypermap(adjacency_list, vertices=None, representation='hypol'):
    if vertices is None:
        vertices = {vertex: {} for vertex in adjacency_list}
    else:
        vertices = deepcopy(vertices)
    m_param = 5
    L_param = 0.01
    gamma = 2.999
    bar_k = 2 * (m_param + L_param)
    beta = 1 / (gamma-1)
    T_param = 0.5

    N = len(adjacency_list)
    vertex_order = sorted(adjacency_list, key=lambda v: len(adjacency_list[v]), reverse=True)
    coords = np.empty((N, 2), dtype=float)
    first_vertex = vertex_order[0]
    coords[0] = 0, 2 * np.pi * np.random.random()
    vertices[first_vertex] = {'r': coords[0, 0], 'phi': coords[0, 1]}
    for idx, vertex in enumerate(vertex_order[1:], start=1):
        t = idx + 1
        r_t = 2 * np.log(t)
        coords[:idx-1] = coords[:idx-1] * beta + (1 - beta) * r_t
        I = (1 - t**(beta-1)) / (1-beta)
        L = 2 * L_param * (1-beta) / (2*beta-1) / (1-N**(beta-1))**2 * ((N/t)**(2*beta-1)-1) * (1-t**(beta-1))
        m = m_param + L
        R = r_t - 2 * np.log(2 * T_param / np.sin(T_param*np.pi) * I / m)
        if R < r_t:
            print('negative R!')
        adjacent = np.array([vertex_order[i] in adjacency_list[vertex] for i in range(idx)], dtype=int)
        
        r_s, theta_s = coords[:idx].T
        best_theta = 0
        theta_t = np.linspace(0, 2*np.pi, int(2 * np.pi * t), endpoint=False)[:,None]
        d_theta = np.pi - abs(np.pi-abs(theta_t - theta_s))
        x_st = np.arccosh(np.cosh(r_s)*np.cosh(r_t)-np.sinh(r_s)*np.sinh(r_t)*np.cos(d_theta))
        P_st = 1 / (1 + np.exp((x_st-R)/(2*T_param)))
        if idx <= m:
            log_L = np.sum(np.log(1 - P_st), 1)
        else:
            log_L = np.sum(np.log(1 - adjacent + adjacent * P_st), 1)
        best_theta = float(theta_t[len(log_L)-1-np.argmax(log_L[::-1])])
        coords[idx] = R, best_theta
        vertices[vertex] = {'r': R, 'phi': best_theta}
    return vertices