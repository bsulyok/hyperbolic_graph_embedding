#from src.mercator import mercator
#from src.hypermap import hypermap
from src.greedy_embedding import greedy_embedding
#from src.ncmce import ncmce
from src.io import read, write
from src import greedy_routing_annealing as gra

if __name__ == "__main__":
	adj, vert = read('test_graphs/pso_512.pickle')
	coord = gra.get_coordinates(adj, vert)
	adjm = gra.get_adjacency_matrix(adj, vert)
	distm = gra.distance_matrix_alt(coord)