import networkx as nx
import pandas as pd
import numpy as np
from operator import itemgetter
import sys
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

def graph_from_df(df):
    G = nx.DiGraph()

    for index, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'])#, weight = row['Weight'])
    
    return G

def get_katz_centrality(G, beta, alpha):
    katz_centrality = []
    for node, katz in nx.katz_centrality(G, alpha = alpha, beta = beta, max_iter = 100000).items():
        katz_centrality.append(katz)#{'node':node,'katz': katz})
    
    return katz_centrality

def get_pagerank(G, alpha, max_iter):
    pagerank = []
    for node, page_rank in nx.pagerank(G, alpha = alpha, max_iter = max_iter).items():
        pagerank.append({'node':node,'pagerank': page_rank})
    
    return pagerank

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def to_symmetric(matrix):
    for i in range(len(matrix)):
        for j in range(i):
            if matrix[i][j] == 1:
                matrix[j][i] = matrix[i][j]
            elif matrix[i][j] == 0:
                matrix[i][j] = matrix[j][i]
            else:
                print("wtf")

def get_in_degree(G):
    degrees = []
    for node, degree in G.in_degree():
        degrees.append(degree)#{'node':node, 'degree':degree})
    
    return degrees

def get_out_degree(G):
    degrees = []
    for node, degree in G.out_degree():
        degrees.append(degree)#{'node':node, 'degree':degree})
    
    return degrees

def get_degree(G):
    degrees = []
    for node, degree in G.degree():
        degrees.append(degree})
    
    return degrees

def get_kth_smallest_nodes(katz, k):
    smallest = []
    for i in range(k):
        smallest.append(katz[i]["node"])

    return smallest

def get_kth_highest_nodes(katz, k):
    highest = []
    n = len(katz)
    for i in range(k):
        highest.append(katz[n - (i + 1)]["node"])

    return highest    

def main():
    dir = "/home/nicolas/Documentos/redes-complexas/medidas-centralidade/data/"
    
    df = pd.read_csv(dir + "otc.csv")
    G = graph_from_df(df)    
    
    katz = get_katz_centrality(G, 1.0, 0.021262409665728472)
    in_degree = get_in_degree(G)
    out_degree = get_out_degree(G)

    print(pearsonr(katz, out_degree), " ", pearsonr(katz, in_degree))
    print(spearmanr(katz, out_degree), " ", spearmanr(katz, in_degree))
    
if __name__ == '__main__':
    main()
