import mercator
from .typing import AdjacencyList, VertexList
from time import time_ns
import csv
import numpy as np
import os

def mercator_embedding(adjacency_list: AdjacencyList) -> VertexList:
	
	# creating temporary edge list
	time_stamp = time_ns()
	with open(str(time_stamp) + '.csv', mode='w', newline='') as tmp:
		csv_writer = csv.writer(tmp, delimiter=' ')
		for vertex, neighbourhood in adjacency_list.items():
			for neighbour in neighbourhood:
				csv_writer.writerow([vertex, neighbour])

	# calling the mercator embedding
	mercator.embed(str(time_stamp) + '.csv', clean_mode=True, quiet_mode=True)

	# retrieving the embedding as hyperboloid coordinates
	coord = np.genfromtxt(str(time_stamp) + '.inf_coord_raw')
	vertex_list = {}
	for idx, vertex in enumerate(adjacency_list):
		vertex_list[vertex] = {'r': coord[idx, 2], 'theta': coord[idx, 1]}
	
	# cleanup
	os.remove(str(time_stamp) + '.csv')
	os.remove(str(time_stamp) + '.inf_coord')
	os.remove(str(time_stamp) + '.inf_coord_raw')

	return vertex_list