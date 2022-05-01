from numpy.core.numeric import full

from graphx import geometry
import numpy as np
from scipy.special import hyp2f1
from copy import deepcopy
from itertools import combinations
from scipy import integrate
from ..topology import global_clustering_coefficient


def mercator(adjacency_list, vertices, **kwargs):
    
    if vertices is None:
        vertices = {vertex: {} for vertex in adjacency_list}
    else:
        vertices = deepcopy(vertices)
    
    np.random.seed(0)

    beta_min, beta_max = 1, -1
    beta_found = False

    acceptable_kappa_deviation = 1e-5
    lcc_sample_count = 600
    acceptable_lcc_deviation = 1e-2

    beta = 2 + np.random.rand()

    N = len(vertices)
    R = N / 2 / np.pi

    vertex_degree = np.array([[vert, len(neigh)] for vert, neigh in adjacency_list.items()], dtype=int)
    vertex, degree = vertex_degree[np.argsort(vertex_degree[:,1])][::-1].T
    inverse_vertex_dict = {vert: idx for idx, vert in enumerate(vertex)}
    degree_class, inverse_degree_class, degree_class_count = np.unique(degree, return_inverse=True, return_counts=True)
    degree_distribution = degree_class_count / np.sum(degree_class_count)
    average_degree = np.average(degree_class, weights=degree_distribution)
    degree_sample_distribution = degree_class * degree_distribution / average_degree

    N_multiedge = len(np.where(degree != 1)[0])
    N_class = len(degree_class)

    def degree_sampler(size):
        return np.random.choice(range(N_class), size=size, p=degree_sample_distribution)

    mu = beta / 2 / np.pi / average_degree * np.sin(np.pi / beta)
    
    def connection_probability(k_1, k_2):
        return hyp2f1(1, 1/beta, 1 + 1/beta, -(R*np.pi/mu/k_1/k_2)**beta)

    def d_theta_sampler(k_1, k_2):
        a = 1 / connection_probability(k_1, k_2) / np.pi
        b = R / mu / k_1 / k_2
        c = beta
        def d_theta_dist(d_theta):
            return a / (1 + (b * d_theta)**c)
        d_theta = np.pi * np.random.rand()
        while d_theta_dist(d_theta) < np.random.rand():
            d_theta = np.pi * np.random.rand()
        return d_theta

    def exp_theta(k_1, k_2):
        a = np.pi * hyp2f1(1, 2/beta, 1 + 2/beta, -(R*np.pi/mu/k_1/k_2)**beta)
        b = 2 * hyp2f1(1, 1/beta, 1 + 1/beta, -(R*np.pi/mu/k_1/k_2)**beta)
        return a / b
    
    observed_lcc = global_clustering_coefficient(adjacency_list)

    while not beta_found:

        print(beta)

        #####
        # 1 #
        #####

        mu = beta / 2 / np.pi / average_degree * np.sin(np.pi / beta)

        #####
        # 2 #
        #####

        def infer_deviation(kappa):
            exp_degree = np.zeros_like(kappa)
            for (i, kappa_i), (j, kappa_j) in combinations(enumerate(kappa), 2):
                p = connection_probability(kappa_i, kappa_j)
                exp_degree[i] += p * degree_class_count[j]
                exp_degree[j] += p * degree_class_count[i]
            for i, kappa_i in enumerate(kappa):
                p = connection_probability(kappa_i, kappa_i)
                exp_degree[i] += p * (degree_class_count[i] - 1)
            return degree_class - exp_degree

        kappa = degree_class.copy().astype(float)

        deviation, largest_deviation = np.zeros_like(kappa), 10 * acceptable_kappa_deviation
        while acceptable_kappa_deviation < largest_deviation:
            kappa = np.abs(kappa + np.random.random(len(kappa)) * deviation)
            deviation = infer_deviation(kappa)
            largest_deviation = np.max(np.abs(deviation))

        #####
        # 3 #
        #####

        exp_lcc = np.zeros_like(degree_class, dtype=float)
        for k, kappa_k in enumerate(kappa):
            for _ in range(lcc_sample_count):
                i, j = degree_sampler(2)
                if 1 < i and 1 < j:
                    kappa_i, kappa_j = kappa[i], kappa[j]
                    d_theta_i = d_theta_sampler(kappa_i, kappa_k)
                    d_theta_j = d_theta_sampler(kappa_j, kappa_k)
                    if np.random.randint(2, dtype=bool):
                        d_theta_ij = abs(d_theta_i - d_theta_j)
                    else:
                        theta_sum = d_theta_i + d_theta_j
                        d_theta_ij = min(theta_sum, 2*np.pi - theta_sum)
                    p = 1 / (1 + (R * d_theta_ij / mu / kappa[i] / kappa[j])**beta)
                    exp_lcc[k] += p
        exp_lcc = exp_lcc / lcc_sample_count
        exp_mean_lcc = np.average(exp_lcc, weights=degree_class_count)

        if abs(exp_mean_lcc - observed_lcc) < acceptable_lcc_deviation:
            beta_found = True
        elif observed_lcc < exp_mean_lcc:
            beta_max = beta
            beta = (beta_max + beta_min) / 2
        elif beta_max == -1:
            beta_min = beta
            beta = 1.5 * beta
        else:
            beta_min = beta
            beta = (beta_max + beta_min) / 2

    kappa = kappa[inverse_degree_class]

    #####
    # 4 #
    #####

    cord_endpoints, cord_length = [], []
    #for vertex_i, vertex_j, attr in utils.edge_iterator(adjacency_list):
    for vertex_i, neighbourhood_i in adjacency_list.items():
        for vertex_j in neighbourhood_i:
            idx_i, idx_j = inverse_vertex_dict[vertex_i], inverse_vertex_dict[vertex_j]
            if vertex_j < vertex_i and 1 < degree[idx_i] and 1 < degree[idx_j]:
                cord_length.append(2 * np.sin(exp_theta(kappa[idx_i], kappa[idx_j])/2))
                cord_endpoints.append([inverse_vertex_dict[vertex_i], inverse_vertex_dict[vertex_j]])


    #####
    # 6 #
    #####
    
    cord_length_variance = np.var(cord_length)
    weight_matrix = np.zeros((N_multiedge, N_multiedge), dtype=float)
    for (idx_i, idx_j), cord_len in zip(cord_endpoints, cord_length):
        weight = np.exp(-cord_len**2/cord_length_variance)
        weight_matrix[idx_i, idx_j] = weight
        weight_matrix[idx_j, idx_i] = weight
    diagonal = np.sum(weight_matrix, axis=0) * np.eye(N_multiedge)
    weighted_laplacian = diagonal - weight_matrix

    eig_val, eig_vec = np.linalg.eig(weighted_laplacian)
    v_1, v_2 = eig_vec[np.argsort(eig_val)[::-1][:2]]

    #####
    # 5 #
    #####

    angular_coord = np.arctan2(v_2, v_1)

    #####
    # 7 #
    #####

    angular_order = np.zeros_like(vertex, dtype=float)
    for idx, vert in enumerate(np.argsort(angular_coord)):
        angular_order[vert] = idx
    for vert in range(N_multiedge, len(vertex)):
        neigh = inverse_vertex_dict[next(iter(adjacency_list[vertex[vert]]))]
        angular_order[vert] = angular_order[neigh] + np.random.rand() - 1 / 2

    arg_sort_angular_order = np.argsort(angular_order)
     
    vertex = vertex[arg_sort_angular_order]
    kappa = kappa[arg_sort_angular_order]
    
    def gap_distribution(g):
        return N / 2 / np.pi * np.exp(-N / 2 / np.pi * g)

    def connected_conditional_connectivity(g, k_1, k_2):
        return 1 / (1 + (R*g/mu/k_1/k_2)**beta)

    def unconnected_conditional_connectivity(g, k_1, k_2):
        return 1 / (1 + (R*g/mu/k_1/k_2)**beta)

    def exp_gap_connected(k_1, k_2):
        def func_1(x):
            return x * connected_conditional_connectivity(x, k_1, k_2) * gap_distribution(x)
        def func_2(x):
            return connected_conditional_connectivity(x, k_1, k_2) * gap_distribution(x)
        y_1, err_1 = integrate.quad(func_1, 0, np.pi)
        y_2, err_2 = integrate.quad(func_2, 0, np.pi)
        return y_1 / y_2

    def exp_gap_unconnected(k_1, k_2):
        def func_1(x):
            return x * unconnected_conditional_connectivity(x, k_1, k_2) * gap_distribution(x)
        def func_2(x):
            return unconnected_conditional_connectivity(x, k_1, k_2) * gap_distribution(x)
        y_1, err_1 = integrate.quad(func_1, 0, np.pi)
        y_2, err_2 = integrate.quad(func_2, 0, np.pi)
        return y_1 / y_2

    exp_gap = np.empty(N, dtype=float)


    for idx, cur_vert in enumerate(vertex):
        prev_vert = arg_sort_angular_order[idx-1]
        cur_kappa = kappa[cur_vert]
        prev_kappa = kappa[prev_vert]
        if prev_vert in adjacency_list[cur_vert]:
            exp_gap[idx] = exp_gap_connected(cur_kappa, prev_kappa)
        else:
            exp_gap[idx] = exp_gap_unconnected(cur_kappa, prev_kappa)

    #####
    # 8 #
    #####

    theta = np.cumsum(2 * np.pi * exp_gap / np.sum(exp_gap))
    kappa_min = np.min(kappa)
    r_max = 2*np.log(N / mu / np.pi / kappa_min**2)

    coord_transform = kwargs.get('coord_transform', geometry.identity)
    for vert_i, kappa_i, theta_i in zip(vertex, kappa, theta):
        r_vertex, theta_vertex = coord_transform(r_max-2*np.log(kappa_i/kappa_min), theta_i)
        vertices[vert_i]= {'r': r_vertex, 'theta': theta_vertex}
    
    return vertices