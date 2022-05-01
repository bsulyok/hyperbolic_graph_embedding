from src.mercator import mercator
from src.hypermap import hypermap
from src.greedy_embedding import greedy_embedding
from src.ncmce import ncmce
from src.io import read, write

if __name__ == "__main__":
	adj, vert = read('test_graphs/pso_64.pickle')