import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import math
import time
#from pygraph.classes.graph import graph

def graph_from_df(df):
    G = nx.DiGraph()

    for index, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight = row['Weight'])
    
    return G

def get_degree(G):
    degrees = []
    for node, degree in G.degree():
        degrees.append(degree)
    
    return degrees

def get_in_degree(G):
    degrees = []
    for node, degree in G.in_degree():
        degrees.append(degree)
    
    return degrees

def get_out_degree(G):
    degrees = []
    for node, degree in G.out_degree():
        degrees.append(degree)
    
    return degrees

def get_katz_centrality(G, alpha):
    katz_centrality = []
    for node, katz in nx.katz_centrality(G, alpha = alpha).items():
        katz_centrality.append(katz)

    return katz_centrality

def get_eigenvector_centrality(G, max_iter):
    eigenvector_centrality = []
    for node, eigen in nx.eigenvector_centrality(G, max_iter = max_iter).items():
        eigenvector_centrality.append(eigen)

    return eigenvector_centrality

def get_accessibility(adjacency_matrix):
    strength = []    
    accessibility = []
    
    for i in range(len(adjacency_matrix)):
        strength.append(0)
        for j in range(len(adjacency_matrix[0])):
            strength[i] += adjacency_matrix[i][j]

    s = []
    for i in range(len(adjacency_matrix)):
        sum = 0
        for j in range(len(adjacency_matrix[0])):
            if adjacency_matrix[i][j] != 0:
                p_i = adjacency_matrix[i][j]/strength[i]
                sum += p_i * math.log(p_i)
        s.append(-1 * sum)
    
    for i in range(len(adjacency_matrix)):
        accessibility.append(math.exp(s[i]))

    return accessibility

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def to_symmetric(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[j][i] = matrix[i][j]
    
def main():
    dir = "/home/nicolas/Documentos/redes-complexas/medidas-redes-complexas/data/"
    
    df = pd.read_csv(dir + "moreno.csv")
    G = graph_from_df(df)
    
    adjacency_matrix = nx.adjacency_matrix(G).toarray()
    to_symmetric(adjacency_matrix)
    
    accessibility = get_accessibility(adjacency_matrix)
    degrees = get_degree(G)

    f, ax = plt.subplots(figsize=(7, 7))
    #ax.set(xscale = 'log', yscale = 'log')
    
    sns.scatterplot(x = degrees, y = accessibility)
    #sns.distplot(a = degrees, ax = ax, hist = False, color = 'black')
    
    for i in range(len(degrees)):
        if degrees[i] == 2:
            print(accessibility[i])

    #plt.xscale(value = "log")
    #plt.yscale(value = "log")
    plt.xlim(right = max(degrees) + 50)
    plt.ylim(top = 160)
    plt.xlabel('Grau')
    plt.ylabel('Acessibilidade')
    plt.plot()
    plt.show()

if __name__ == '__main__':
    main()