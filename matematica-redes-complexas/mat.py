import numpy as np
import pandas as pd
import time
import networkx as nx
import igraph as ig

def count_nonzero(adjacency_matrix):
    print("Mult x1...")
    X = np.matmul(adjacency_matrix, adjacency_matrix)
    print("Mult x2...")
    X = np.matmul(X, adjacency_matrix)

    print("Mult y1...")
    Y = np.matmul(adjacency_matrix, adjacency_matrix)
    print("Mult y2...")
    Y = np.matmul(Y, adjacency_matrix)
    print("Mult y3...")
    Y = np.matmul(Y, adjacency_matrix)

    print("Counting nonzero in X")
    X_nonzero = np.count_nonzero(X)
    print("Counting nonzero in Y")
    Y_nonzero = np.count_nonzero(Y)

    print("X ", X_nonzero)
    print("Y ", Y_nonzero)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def main():
    df = pd.read_csv("/home/nicolas/Documentos/redes-complexas/matematica-redes-complexas/data/rede1.csv")
    G = nx.read_pajek("/home/nicolas/Documentos/redes-complexas/matematica-redes-complexas/data/rede3.paj")
    #G = nx.DiGraph()
    #for index, row in df.iterrows():
        #source = (row['Source'], {'weight': row['Weight']})
        #target = (row['Target'], {'weight': row['Weight']})
        #G.add_edge(source, target)
    for a, b, c in G.edges(data=True):
        print(a, " ", b, " ", c)

    #print("Building adj matrix...")
    #adjacency_matrix = nx.adjacency_matrix(G).toarray()
    
    #print(len(adjacency_matrix))
    #print(len(adjacency_matrix[0]))
    
    #for i in range(len(adjacency_matrix)):
    #    print("")
    #    for j in range(len(adjacency_matrix[0])):
    #        print(adjacency_matrix[i][j], " ", end = "")

    #print(check_symmetric(adjacency_matrix))

    #print("Counting non zeros...")
    #count_nonzero(adjacency_matrix)
    
    #print(list(G.nodes(data=True)))

    print(nx.number_of_nodes(G))
    print(nx.number_of_edges(G))
    #reciprocity = nx.reciprocity(G)
    #print(reciprocity)

if __name__ == '__main__':
    main()