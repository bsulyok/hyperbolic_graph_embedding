import pickle

def write(data, file_name):
	with open(file_name, mode='wb') as outfile:
		pickle.dump(data, outfile)


def read(file_name):
	with open(file_name, mode='rb') as infile:
		return pickle.load(infile)