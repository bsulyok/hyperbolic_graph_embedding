from numpy.core.numeric import full

from graphx import geometry
from ... import utils
from ...utils.common import disjoint_set
from ...utils.distance import distance
import numpy as np
from scipy.special import zeta, hyp2f1
from scipy.stats import rv_continuous
from copy import deepcopy
from tqdm import tqdm
from itertools import combinations
from scipy import integrate
from ...utils.iterator import edge_iterator

def minimum_weight_spanning_tree(adjacency_list, weight_attribute='weight'):
    '''
    Find the minimum weight spanning tree rooted at root of the provided graph.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    weight_attribute : str
        The weight attribute key that will be checked when extracting edge data.

    Returns
    -------
    tree_adjacency_list : dict of dicts
        The adjacency list of the minimum weight spanning tree.
    '''
    N = len(adjacency_list)
    tree_adjacency_list = {vertex: {} for vertex in adjacency_list}
    disjoint_vertices = disjoint_set(N)
    edge_queue = sorted(edge_iterator(adjacency_list, True), key=lambda li: li[2][weight_attribute])
    component_size = 1
    for vertex, neighbour, attributes in edge_queue:
        if not disjoint_vertices.is_connected(vertex, neighbour):
            disjoint_vertices.union(vertex, neighbour)
            tree_adjacency_list[vertex].update({neighbour: adjacency_list[vertex][neighbour]})
            tree_adjacency_list[neighbour].update({vertex: adjacency_list[neighbour][vertex]})
            component_size = max(component_size, disjoint_vertices.size(vertex), disjoint_vertices.size(neighbour))
        if component_size == N:
            break
    return tree_adjacency_list


def calculate_CCDF(degree):
    min_degree, max_degree = min(degree), max(degree)
    degree_values = np.arange(min_degree, max_degree + 1)
    degree_distribution = np.zeros_like(degree_values)
    for deg in degree:
        degree_distribution[deg - min_degree] += 1
    return degree_values, np.cumsum(degree_distribution[::-1])[::-1]


def estimate_beta(degree, min_samples=50):
    degree_values = np.unique(degree)
    min_degree, max_degree = min(degree_values), max(degree_values)
    degree_axis = np.arange(min_degree, max_degree + 1)
    degree_distribution = np.zeros_like(degree_axis)
    for deg in degree:
        degree_distribution[deg - min_degree] += 1
    CCDF = np.cumsum(degree_distribution[::-1])[::-1]
    gamma, D = None, None
    for deg_min in degree_values:
        proper_degree_list = degree[deg_min <= degree]
        if min_samples <= len(proper_degree_list):
            new_gamma = 1 + 1/np.mean(np.log(proper_degree_list/(deg_min-1/2)))
            deg_min_idx = np.where(degree_axis == deg_min)[0][0]
            degrees_in_tail = degree_axis[deg_min_idx:]
            distribution_in_tail = CCDF[deg_min_idx:]
            tmp_1 = np.log(distribution_in_tail)
            tmp_2 = (new_gamma-1) * np.log(degrees_in_tail)
            const = np.exp(np.mean(tmp_1 + tmp_2)) * (new_gamma-1)
            Z = zeta(new_gamma, deg_min)
            tmp_1 = distribution_in_tail / const / Z
            tmp_2 = degrees_in_tail ** (1-new_gamma) / (new_gamma-1) / Z
            new_D = np.max(np.abs(tmp_1 - tmp_2))
            if D is None or new_D < D:
                gamma, D = new_gamma, new_D
        else:
            break
    return 1 / (gamma-1)


def pre_weighting(adjacency_list):
    degree = {}
    for vertex, neighbourhood in adjacency_list.items():
        degree[vertex] = len(neighbourhood)
    for vertex, neighbour, attributes in edge_iterator(adjacency_list, True):
        common_neighbours = set(adjacency_list[vertex].keys()).intersection(
            set(adjacency_list[neighbour].keys())
        )
        tmp_1 = degree[vertex] + degree[neighbour]
        tmp_2 = degree[vertex] * degree[neighbour]
        weight = (tmp_1 + tmp_2) / (1 + len(common_neighbours))
        adjacency_list[vertex][neighbour].update({'weight': weight})
    return adjacency_list


def circular_adjustment(angular_coordinates):
    tmp = 2 * np.pi * (angular_coordinates - min(angular_coordinates))
    return tmp / max(angular_coordinates)-min(angular_coordinates)


def equidistant_adjustment(angular_coordinates):
    d_phi = 2 * np.pi / len(angular_coordinates)
    phi = np.empty_like(angular_coordinates)
    for idx, vertex in enumerate(np.argsort(angular_coordinates)):
        phi[vertex] = idx * d_phi
    return phi


def ncMCE(adjacency_list, vertices=None, adjustment='equidistant', metric_space='native_disk'):
    if vertices is None:
        vertices = {vertex: {} for vertex in adjacency_list}
    else:
        vertices = deepcopy(vertices)

    if adjustment == 'equidistant':
        angular_adjustment = equidistant_adjustment
    elif adjustment == 'circular':
        angular_adjustment = circular_adjustment

    weighted_adjacency_list = pre_weighting(deepcopy(adjacency_list))
    min_tree_adjacency_list = minimum_weight_spanning_tree(
        adjacency_list=weighted_adjacency_list
    )
    vertex_distance = utils.distance(min_tree_adjacency_list)

    D = []
    for vertex in min_tree_adjacency_list:
        d = []
        for neighbour in min_tree_adjacency_list:
            d.append(vertex_distance[vertex][neighbour])
        D.append(d)
    D = np.array(D)
    U, S, VH = np.linalg.svd(D, full_matrices=False)
    S[2:] = 0
    coordinates = np.transpose(np.sqrt(np.diag(S)) @ VH)
    angular_coordinates = angular_adjustment(coordinates[:, 1])
    degree = []
    for neighbourhood in adjacency_list.values():
        degree.append(len(neighbourhood))
    degree = np.array(degree)
    beta = estimate_beta(degree)
    radial_coordinates = np.empty_like(angular_coordinates)
    tmp_1 = np.log(np.linspace(1, len(adjacency_list), len(adjacency_list)))
    tmp_2 = beta * tmp_1 + (1-beta)*np.log(len(adjacency_list))
    radial_coordinates[np.argsort(degree)[::-1]] = 2 * tmp_2

    coord_transform = getattr(geometry.native_disk, f'to_{metric_space}')

    for idx, vertex in enumerate(vertices):
        r, theta = coord_transform(radial_coordinates[idx],angular_coordinates[idx])
        vertices[vertex].update({'r': r, 'theta': theta})
    return vertices