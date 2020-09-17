import numpy as np
import pandas as pd
import time
import networkx as nx
import igraph as ig

def get_undirected_adjacency_matrix(df):
    max_value = df['Source'].max()
    if df['Target'].max() > max_value:
        max_value = df['Target'].max()
   
    adjacency_matrix = np.zeros((max_value, max_value))
    for index, row in df.iterrows():
        adjacency_matrix[row['Source'] - 1][row['Target'] - 1] = 1
        adjacency_matrix[row['Target'] - 1][row['Source'] - 1] = 1
    
    return adjacency_matrix

def get_directed_adjacency_matrix(df):
    max_value = df['Source'].max()
    if df['Target'].max() > max_value:
        max_value = df['Target'].max()

    adjacency_matrix = np.zeros((max_value, max_value))
    for index, row in df.iterrows():
        adjacency_matrix[row['Source'] - 1][row['Target'] - 1] = 1

    return adjacency_matrix

def count_nonzero(adjacency_matrix):
    X = np.matmul(adjacency_matrix, adjacency_matrix)
    X = np.matmul(X, adjacency_matrix)
    print(len(X))
    print(len(X[0]))

    Y = np.matmul(adjacency_matrix, adjacency_matrix)
    Y = np.matmul(Y, adjacency_matrix)
    Y = np.matmul(Y, adjacency_matrix)
    print(len(Y))
    print(len(Y[0]))

    X_nonzero = np.count_nonzero(X)
    Y_nonzero = np.count_nonzero(Y)

    print(X_nonzero)
    print(Y_nonzero)

def main():
    df = pd.read_csv("/home/nicolas/Documentos/redes-complexas/matematica-redes-complexas/data/rede2.csv")
    start = time.time()
    
    #list = []
    #for index, row in df.iterrows():
    #    list.append(row['Source'])
    #    list.append(row['Target'])

    #print(len(np.unique(list)))
    
    undirected_adjacency_matrix = get_undirected_adjacency_matrix(df)
    count_nonzero(undirected_adjacency_matrix)    
    
    #directed_adjacency_matrix = get_directed_adjacency_matrix(df)
    #transposed_dir_adjacency_matrix = np.transpose(directed_adjacency_matrix)
    #cocitation_matrix = np.matmul(directed_adjacency_matrix, transposed_dir_adjacency_matrix)
    #coupling_matrix = np.matmul(transposed_dir_adjacency_matrix, directed_adjacency_matrix)

    #strongest_coupling = 0
    #strongest_cocitation = 0
    #strongest_index = -1
    #for i in range(len(cocitation_matrix)):
    #    if cocitation_matrix[i][i] > strongest_cocitation and coupling_matrix[i][i] > strongest_coupling:
    #        strongest_cocitation = cocitation_matrix[i][i]
    #        strongest_coupling = coupling_matrix[i][i]
    #        strongest_index = i

    #print(strongest_index)
    #print(strongest_coupling)
    #print(strongest_cocitation)

    #G = nx.DiGraph()

    #for index, row in df.iterrows():
    #    G.add_edge(row['Source'], row['Target'])
    #print(nx.number_of_nodes(G))
    #reciprocity = nx.reciprocity(G)
    #print(reciprocity)

    #g = ig.Graph()
    #g.add_vertex(6005)
    #for index, row in df.iterrows():
    #    g.add_edge(row['Source'], row['Target'])

    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()