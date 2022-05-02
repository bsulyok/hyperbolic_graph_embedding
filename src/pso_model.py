import numpy as np
from math import sin, pi, log, exp
from .typing import AdjacencyList, VertexList, Tuple

def popularity_similarity_optimisation_model(
    N: int,
    m: int,
    beta: float = 0.5,
    T: float = 0,
    **kwargs
) -> Tuple[AdjacencyList, VertexList]:

    def connection_probability(r):
        return 1 / (1+np.exp(r/2/T))

    if beta == 0:
        def cutoff_function(r):
            return r - 2*np.log(T/np.sin(T*np.pi)*(1-np.exp(-r/2))/m)
    elif beta == 1:
        def cutoff_function(r):
            return r - 2*np.log(T/np.sin(T*np.pi)*r/m)
    else:
        def cutoff_function(r):
            return r - 2*np.log(T/np.sin(T*np.pi)*(1-np.exp(-r*(1-beta)/2))/m/(1-beta))
    
    adjacency_list = {v:{} for v in range(N)}
    vertex_list = {v:{} for v in range(N)}

    radial_coord = 2 * np.log(np.arange(1, N+1))
    angular_coord = 2 * pi * np.random.random(N)

    for idx_new in range(N):
        final_radial_coord = beta * radial_coord[-1] + (1-beta) * radial_coord[idx_new]
        r_vertex, theta_vertex = final_radial_coord, angular_coord[idx_new]
        vertex_list[idx_new] = dict(r=r_vertex, theta=theta_vertex)

        if idx_new <= m:
            neighbours = np.arange(idx_new)
        else:
            r_old = beta * radial_coord[idx_new] + (1-beta) * radial_coord[:idx_new]
            theta_old = angular_coord[:idx_new]
            r_new = radial_coord[idx_new]
            theta_new = angular_coord[idx_new]
            with np.errstate(divide='ignore', invalid='ignore'):
                ang_diff = np.pi - np.abs(np.pi - np.abs(theta_new - theta_old))
                vertex_distance = np.cosh(r_new) * np.cosh(r_old)
                vertex_distance -= np.sinh(r_new) * np.sinh(r_old) * np.cos(ang_diff)
                vertex_distance = np.arccosh(vertex_distance)
            if T == 0:
                neighbours = sorted(range(idx_new), key=lambda idx_old: vertex_distance[idx_old])[:m]
            else:
                cutoff = cutoff_function(r_old)
                conn_prob = connection_probability(vertex_distance-cutoff)
                neighbours = np.random.choice(idx_new, m, replace=False, p=conn_prob/sum(conn_prob))
        for idx_old in neighbours:
            adjacency_list[idx_new][idx_old] = {}
            adjacency_list[idx_old][idx_new] = {}

    return adjacency_list, vertex_list


PSO = popularity_similarity_optimisation_model