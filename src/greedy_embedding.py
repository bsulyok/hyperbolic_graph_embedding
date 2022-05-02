from .utils import breadth_first_distance, minimum_depth_spanning_tree
import numpy as np
from copy import deepcopy
from .typing import AdjacencyList, VertexList


def get_moebius_transformation(z1, w1, z2, w2, z3, w3):
    a = np.linalg.det(np.array([
        [z1*w1, w1, 1],
        [z2*w2, w2, 1],
        [z3*w3, w3, 1]
    ]))
    b = np.linalg.det(np.array([
        [z1*w1, z1, w1],
        [z2*w2, z2, w2],
        [z3*w3, z3, w3]
    ]))
    c = np.linalg.det(np.array([
        [z1, w1, 1],
        [z2, w2, 1],
        [z3, w3, 1]
    ]))
    d = np.linalg.det(np.array([
        [z1*w1, z1, 1],
        [z2*w2, z2, 1],
        [z3*w3, z3, 1]
    ]))
    return np.array([[a, b], [c, d]])


def invert_mobius(M):
    return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])


def apply_mobius_transformation(z, M):
    return (M[0, 0] * z + M[0, 1]) / (M[1, 0] * z + M[1, 1])


def greedy_embedding(
    adjacency_list: AdjacencyList
):
    '''
    This is a hyperbolic graph embedding algorithm written following the work outlined by R. Kleinberg in their 2007 publication.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.

    Returns
    -------
    vertex_list : dict of dicts
        Vertex list containing the inferred hyperbolic coordinates.
    '''

    vertex_list = {vertex: {} for vertex in adjacency_list}

    root, min_max_dist = 0, np.inf
    for vertex in adjacency_list:
        dist = breadth_first_distance(adjacency_list, vertex)
        max_dist = max(dist.values())
        if max_dist < min_max_dist:
            root, min_max_dist = vertex, max_dist

    children = minimum_depth_spanning_tree(adjacency_list, root)
    tree_degree = max(len(ch) for ch in children.values()) + 1

    # transform the final positions to the native disk
    coord_transform = lambda r:  2*np.arctanh(r)

    # initialize the relevant Moebius transformations
    last_root = np.exp(-2j*np.pi/tree_degree)
    last_half_root = np.exp(-2j*np.pi/2/tree_degree)
    sigma = get_moebius_transformation(1, 1, last_root, -1, last_half_root, -1j)
    isigma = get_moebius_transformation(1, 1, -1, last_root, -1j, last_half_root)
    A = np.array([[-1, 0], [0, 1]])
    B = []
    for k in range(tree_degree):
        tmp = np.array([[np.exp(2j*np.pi*k/tree_degree), 0], [0, 1]])
        B.append(sigma @ tmp @ isigma)

    # fix the origin
    u = apply_mobius_transformation(0j, sigma)
    v = u.conjugate()
    root_transform = sigma

    def iterative_coordination_search(vertex, transform):
        coord = apply_mobius_transformation(v, invert_mobius(transform))
        r = abs(coord)
        r = coord_transform(r)
        angle = np.arctan2(coord.imag, coord.real)
        vertex_list[vertex].update({'r': r, 'theta': angle})
        for child_id, child in enumerate(children[vertex], 1):
            child_transform = B[child_id] @ A @ transform
            iterative_coordination_search(child, child_transform)

    vertex_list[root].update({'r': 0, 'theta': 0})
    for child_id, child in enumerate(children[root]):
        child_transform = B[child_id] @ root_transform
        iterative_coordination_search(child, child_transform)
    
    return vertex_list