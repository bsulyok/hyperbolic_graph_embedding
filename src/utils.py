import queue
from .typing import AdjacencyList, VertexList
import plotly.graph_objects as go
import numpy as np
import heapq
from operator import itemgetter
from copy import deepcopy


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

def pre_weighting(
    adjacency_list: AdjacencyList
) -> AdjacencyList:

    neighbour_set = {v: set(neigh) for v, neigh in adjacency_list.items()}
    for vertex, neighbourhood in adjacency_list.items():
        for neighbour in neighbourhood:
            weight = len(neighbour_set[vertex]) + len(neighbour_set[neighbour])
            weight /= (len(set.intersection(neighbour_set[vertex], neighbour_set[neighbour])) + 1) 
            adjacency_list[vertex][neighbour]['weight'] = weight
    return adjacency_list


def dijsktra(
    adjacency_list: AdjacencyList,
    vertex_list: VertexList,
    weight_attribute='weight'
) -> np.ndarray:

    N = len(adjacency_list)
    vertex_dict = {vertex: idx for idx, vertex in enumerate(vertex_list)}
    dist_mat = np.full((N, N), np.inf, dtype=float)
    np.fill_diagonal(dist_mat, 0)
    for source in adjacency_list:
        processed = []
        vertex_queue = [(0, source)]
        heapq.heapify(vertex_queue)
        while vertex_queue:
            vertex_dist, vertex = heapq.heappop(vertex_queue)
            if vertex not in processed:
                processed.append(vertex)
                for neighbour, attributes in adjacency_list[vertex].items():
                    if neighbour not in processed:
                        neighbour_dist = vertex_dist + attributes.get(weight_attribute, 1)
                        if neighbour_dist < dist_mat[vertex_dict[source], vertex_dict[neighbour]]:
                            dist_mat[vertex_dict[source], vertex_dict[neighbour]] = neighbour_dist
                            heapq.heappush(vertex_queue, (neighbour_dist, neighbour))

    return dist_mat


def breadth_first_search(
    adjacency_list: AdjacencyList,
    vertex_list: VertexList
) -> np.ndarray:

    N = len(adjacency_list)
    dist_mat = np.zeros((N, N), dtype=int)
    vertex_dict = {vertex: idx for idx, vertex in enumerate(vertex_list)}
    for source in adjacency_list:
        visited = [source]
        vertex_queue = queue.Queue()
        vertex_queue.put(source)
        dist = {v: np.inf for v in adjacency_list}
        dist[source] = 0
        while not vertex_queue.empty():
            vertex = vertex_queue.get()
            for neighbour in adjacency_list[vertex]:
                if neighbour not in visited:
                    dist[neighbour] = dist[vertex] + 1
                    vertex_queue.put(neighbour)
                    visited.append(neighbour)
        for vertex, d in dist.items():
            dist_mat[vertex_dict[source], vertex_dict[vertex]] = d
    return dist_mat



def get_coordinates(adjacency_list: AdjacencyList, vertex_list: VertexList) -> np.ndarray:    
    coord_getter = itemgetter('r', 'theta')
    return np.array([coord_getter(attr) for attr in vertex_list.values()])


def distance_matrix(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        part_0 = np.cosh(x[:, 0])*np.cosh(x[:, np.newaxis, 0])
        part_1 = np.sinh(x[:, 0])*np.sinh(x[:, np.newaxis, 0])
        ang_diff = np.cos(x[:, 1]*x[:, np.newaxis, 1])
        dist_mat = np.arccosh(part_0 - part_1 * ang_diff)
        dist_mat = np.arccosh(dist_mat)
        np.fill_diagonal(dist_mat, 0)
    return dist_mat


def mapping_accuracy(
    adjacency_list: AdjacencyList,
    vertex_list: VertexList
) -> np.ndarray:
    
    #adjacency_list = pre_weighting(deepcopy(adjacency_list))
    graph_distance = dijsktra(adjacency_list, vertex_list)
    metric_distance = distance_matrix(get_coordinates(adjacency_list, vertex_list))

    indices = np.triu_indices(len(graph_distance), k=1)
    x = graph_distance[indices]
    y = metric_distance[indices]
    
    corrcoeff = np.corrcoef(x, y)[0, 1]
    
    fig = go.Figure(
        go.Scattergl(
            x = x,
            y = y,
            mode='markers'
        )
    )

    fig.update_layout(
        title=f'Correlation coefficient: {corrcoeff}',
        xaxis_title='graph distance',
        yaxis_title='metric distance',
        width=1920,
        height=1080
    )


    fig.write_image(f'images/non_weighted_{len(adjacency_list)}.png')
    return corrcoeff


'''
def mapping_accuracy(graph_distance: np.ndarray, metric_distance: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(len(graph_distance), k=1)
    a = graph_distance[indices]
    b = metric_distance[indices]
    return np.corrcoef(a, b)
'''


def global_clustering_coefficient(
    adjacency_list: AdjacencyList
) -> VertexList:
    triangles, triplets = 0, 0
    for neighbourhood in adjacency_list.values():
        degree = len(neighbourhood)
        if degree <= 1:
            continue
        for neigh_1, neigh_2 in combinations(neighbourhood, 2):
            if neigh_2 in adjacency_list[neigh_1]:
                triangles += 1
        triplets += degree * (degree - 1)
    triplets = triplets / 3
    return triangles / triplets


def breadth_first_distance(adjacency_list: AdjacencyList, source: int) -> VertexList:
    vertex_distance = {}
    vertex_queue = queue.Queue()
    vertex_queue.put(source)
    visited = [source]
    dist = {source: 0}
    while not vertex_queue.empty():
        vertex = vertex_queue.get()
        for neighbour in adjacency_list[vertex]:
            if neighbour not in visited:
                dist[neighbour] = dist[vertex] + 1
                vertex_queue.put(neighbour)
                visited.append(neighbour)
    return dist


def minimum_depth_spanning_tree(
    adjacency_list: AdjacencyList,
    root: int,
    directed: bool = True
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