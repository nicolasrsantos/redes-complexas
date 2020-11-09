import math
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

def graph_from_df(df):
    G = nx.Graph()

    for index, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'])
    
    return G

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_degrees(G):
    degrees = []
    for node, degree in G.degree():
        degrees.append(degree)
    
    return degrees

def plot_histogram(sim):
    f, ax = plt.subplots(figsize=(7, 7))
    sns.set_theme()
    ax = sns.distplot(sim, kde = False, norm_hist = True)
    plt.xlabel('Similaridade')
    plt.ylabel('Distribuição')
    plt.plot()
    plt.show()

# passar matriz A^2, A^3...ao invés de A
def cosine_similarity(A, K):
    similarities = list()
    for i in range(len(A)):
        for j in range(len(A[0])):
            cos_ij = A[i][j] / math.sqrt(K[i] * K[j])
            similarities.append(cos_ij)

    return similarities

def regular_similarity(A, alpha, m):
    sim = 0
    for i in range(m):
        sim += np.dot(alpha, A) ** i
    
    return sim

def main():
    df = pd.read_csv("turker_network.csv")
    G = graph_from_df(df)
    x = dict([(n,0) for n in G])
    print(len(x), len(G.nodes))
    #K = get_degrees(G)
    #A = nx.adjacency_matrix(G).toarray()
    #a = regular_similarity(A, 0.001, 1000)
    #A_power = np.linalg.matrix_power(A, 2)
    #cos_sim = cosine_similarity(A_power, K)
    #plot_histogram(a)

if __name__ == '__main__':
    main()