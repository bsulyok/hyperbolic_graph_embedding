from ...utils import distance
import numpy as np
from copy import deepcopy
from ...utils.typing import Adjacency_list, Vertices
import queue


def minimum_depth_spanning_tree(
    adjacency_list: Adjacency_list,
    root: int = None,
    directed: bool = False
):
    '''
    Find the minimum depth spanning tree rooted at root of the provided graph.

    Parameters
    ----------
    adjacency_list : dict of dicts
        Adjacency list containing edge data.
    root : int
        The vertex at which the tree is rooted.
    directed : bool
        Whether to return a directed or undirected tree. In the undirected case edges originate from parents and point to children.

    Returns
    -------
    tree_adjacency_list : dict of dicts
        The adjacency list of the minimum depth spanning tree.
    '''
    N = len(adjacency_list)
    tree_adjacency_list = {vertex: {} for vertex in adjacency_list}
    vertex_queue = queue.Queue()
    vertex_queue.put(root)
    visited = dict.fromkeys(adjacency_list, False)
    visited[root] = True
    visited_total = 0
    queue_size = 1
    while 0 < queue_size and visited_total < N:
        vertex = vertex_queue.get()
        queue_size -= 1
        for neighbour in adjacency_list[vertex]:
            if not visited[neighbour]:
                tree_adjacency_list[vertex].update({neighbour: adjacency_list[vertex][neighbour]})
                if not directed:
                    tree_adjacency_list[neighbour].update({vertex: adjacency_list[neighbour][vertex]})
                vertex_queue.put(neighbour)
                queue_size += 1
                visited[neighbour] = True
                visited_total += 1
    return tree_adjacency_list


def extract_moebius_transformation(z1, w1, z2, w2, z3, w3):
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
    adjacency_list: Adjacency_list,
    vertices: Vertices = None
):
    if vertices is None:
        vertices = {vertex: {} for vertex in adjacency_list}
    else:
        vertices = deepcopy(vertices)

    min_dist = {}
    for vertex, dist in distance(adjacency_list).items():
        min_dist[vertex] = min(dist.values())
    root = min(min_dist, key=lambda vertex: min_dist[vertex])

    children = minimum_depth_spanning_tree(
        adjacency_list,
        root=root,
        directed=True
    )
    tree_degree = max(len(ch) for ch in children.values()) + 1

    # declare the relevant Mobius transformations
    last_root = np.exp(-2j*np.pi/tree_degree)
    last_half_root = np.exp(-2j*np.pi/2/tree_degree)
    sigma = extract_moebius_transformation(
        1, 1,
        last_root, -1,
        last_half_root, -1j
    )
    isigma = extract_moebius_transformation(
        1, 1,
        -1, last_root,
        -1j, last_half_root
    )
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
        r = 2*np.arctanh(r)
        angle = np.arctan2(coord.imag, coord.real)
        vertices[vertex].update({'r': r, 'theta': angle})
        for child_id, child in enumerate(children[vertex], 1):
            child_transform = B[child_id] @ A @ transform
            iterative_coordination_search(child, child_transform)

    vertices[root].update({'r': 0, 'theta': 0})
    # iteratively find the coordinates of all vertices
    for child_id, child in enumerate(children[root]):
        child_transform = B[child_id] @ root_transform
        iterative_coordination_search(child, child_transform)
    return vertices