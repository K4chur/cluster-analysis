#!/usr/bin/python3
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import isspmatrix
import sklearn.preprocessing

np.set_printoptions(suppress=True, formatter={'float': '{:0.3}'.format})


def calculateP(A):
    N = len(A)
    P = np.zeros((N, N))
    # Calculate the transition probabilities
    for i in range(N):
        row_sum = A[i].sum()
        for j in range(N):
            P[i, j] = A[i, j] / row_sum
    return P


def normalize(matrix):
    return sklearn.preprocessing.normalize(matrix, norm="l1", axis=0)

def expand(P, s):
    if isspmatrix(P):
        return P ** s
    return np.linalg.matrix_power(P, s)

def inflate(P, r):
    n = len(P)
    for i in range(n):
        for j in range(n):
            P[i][j] = np.power(P[i][j], r)

    # Normalize each row of the inflated matrix
    for i in range(n):
        row_sum = sum(P[i])
        for j in range(n):
            P[i][j] /= row_sum
    return P

def sparse_allclose(a, b, rtol=1e-5, atol=1e-8):
    c = np.abs(a - b) - rtol * np.abs(b)
    return c.max() <= atol

def converged(matrix1, matrix2):
    if isspmatrix(matrix1) or isspmatrix(matrix2):
        return sparse_allclose(matrix1, matrix2)

    return np.allclose(matrix1, matrix2)

N = 20
G = nx.Graph()
G.add_nodes_from(range(N))

# Кількість кластерів
k = 10

# Створення списку для зберігання кластерів
clusters = [[] for _ in range(k)]

# Розподіл вершин між кластерами
for i in range(N):
    cluster_index = random.randint(0, k - 1)
    clusters[cluster_index].append(i)

# Додавання ребер з вагами відповідно до кластерів
for cluster in clusters:
    for i in cluster:
        for j in cluster:
            if i != j:
                weight = random.randint(10, 20)
                G.add_edge(i, j, weight=weight)
    for other_cluster in clusters:
        if other_cluster != cluster:
            for i in cluster:
                for j in other_cluster:
                    weight = random.randint(1, 3)
                    G.add_edge(i, j, weight=weight)


# Create a list of colors for each cluster
cluster_colors = ['#FF0000', '#00FF00', '#0000FF']  # You can customize these colors

# Create a mapping of nodes to their respective cluster
node_to_cluster = {}
for i, cluster in enumerate(clusters):
    for node in cluster:
        node_to_cluster[node] = i

# Create a list of node colors based on their cluster
#node_colors = [cluster_colors[node_to_cluster[node]] for node in G.nodes()]

# Create a layout for the graph (e.g., spring layout)
pos = nx.spring_layout(G)

# # Draw the graph with nodes colored by clusters
# nx.draw(G, node_color=node_colors, with_labels=True, node_size=300)
#
# # Display the graph
# plt.show()


#1.2
A = nx.to_numpy_array(G)
P = calculateP(A)
print("P: ")
print(P)

# Calculate the eigenvalues and eigenvectors of P
eigenvalues, eigenvectors = np.linalg.eig(P)

# Print the eigenvalues
print("Eigenvalues of P:")
print(eigenvalues)


###
##TEST G
# G = nx.Graph()
# G.add_nodes_from(range(10))
# G.add_edge(0, 1, weight=4)
# G.add_edge(0, 2, weight=4)
# G.add_edge(1, 2, weight=2)
# G.add_edge(1, 4, weight=1)
# G.add_edge(2, 3, weight=1)
# G.add_edge(3, 5, weight=2)
# G.add_edge(3, 6, weight=1)
# G.add_edge(4, 5, weight=4)
# G.add_edge(6, 7, weight=3)
# G.add_edge(7, 8, weight=3)
# G.add_edge(7, 9, weight=2)
# G.add_edge(8, 9, weight=4)
# k = 3

nx.draw(G, with_labels=True, node_size=300)
plt.show()

# 1.3
def task13(G, k = 3):
    while nx.number_connected_components(G) <= k:
        # Calculate betweenness centrality for all edges
        edge_betweenness = nx.edge_betweenness_centrality(G)
        # Find the edge with the maximum centrality using a list comprehension
        max_edge, max_centrality = max(edge_betweenness.items(), key=lambda x: x[1])

        # Print the edge with the maximum centrality
        print(f"Edge {max_edge}: {max_centrality}")
        G.remove_edge(*max_edge)
        #nx.draw(G, with_labels=True, node_size=300)
        #plt.show()

        num_subgraphs = nx.number_connected_components(G)
        print(f"Clusters now: {num_subgraphs}")
    nx.draw(G, with_labels=True, node_size=300)
    plt.show()

#task13(G, k)

#1.4
def task14(G, s, r):
    A = nx.to_numpy_array(G)
    print("A:")
    print(A)
    Pprev = calculateP(A)
    P = np.copy(Pprev)
    #print(f"P stock:\n{P}")
    i = 0
    max_iterations = 50
    while i < max_iterations:
        print(f"i: {i}, r = {r}, s = {s}")
        P = expand(Pprev, s)
        #print(f"P expanded:\n{P}")
        P = inflate(P, r)
        #print(f"P inflated:\n{P}")
        if converged(P, Pprev):
            break
        Pprev = np.copy(P)
        i += 1

    threshold = 1e-15
    # Replace small values with exact zeros
    P[np.abs(P) < threshold] = 0.0

    print(f"P: {P}")
    # Determine clusters based on the final P matrix
    clusters = []
    for i in range(len(P)):
        for j in range(i, len(P)):
            max_prob = max(P[i, j], P[j, i])
            if i != j and max_prob > 0:
                print(f"i: {i}, j: {j}")
                clusters.append((i, j))

    # Create subgraphs for each cluster
    combined_graph = nx.Graph()
    for cluster in clusters:
        i, j = cluster
        combined_graph.add_edge(i, j)

    # Check if any nodes were left out
    remaining_nodes = set(G.nodes) - set(combined_graph.nodes)
    for node in remaining_nodes:
        combined_graph.add_node(node)

    return combined_graph

#
# Get clusters as a combined graph
combined_graph = task14(G, 2, 2)
print(combined_graph)
print(f"Number of clusters: {nx.number_connected_components(combined_graph)}")
# Plot the combined graph
plt.figure(figsize=(10, 10))
plt.title("All Clusters")
pos = nx.spring_layout(combined_graph)  # Choose a layout that works for your data
nx.draw(combined_graph, pos, with_labels=True)
plt.show()

r_values = [2, 3, 5]  # Fixed values of r
s_values = list(range(2, 11))  # s values from 2 to 10

# For each value of r, calculate the number of clusters for different s values and plot
for r in r_values:
    k_values = []
    for s in s_values:
        combined_graph = task14(G, s, r)
        k = nx.number_connected_components(combined_graph)
        print(f"k = {k}, r = {r}, s = {s}")
        k_values.append(k)

    # Plot the results
    plt.plot(s_values, k_values, label=f'r={r}')

plt.xlabel('s')
plt.ylabel('Number of Clusters')
plt.legend()
plt.grid(True)
plt.show()

#2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_kernels
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

data = pd.read_csv("train.csv")
text_data = data["text"].tolist()



stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def kullback_leibler_divergence(p, q):
    p = np.maximum(p, 1e-10)  # Ensure no division by zero
    q = np.maximum(q, 1e-10)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

preprocessed_text = [preprocess_text(text) for text in text_data]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_text)

kmeans = KMeans(n_clusters=2)
kmeans.fit(tfidf_matrix)

# Отримайте центри кластерів
cluster_centers = kmeans.cluster_centers_

# Ініціалізуйте масив для зберігання відстаней Кульбака–Лейблера
kldistances = np.zeros((tfidf_matrix.shape[0], 2))

# Розрахунок відстаней для кожного зразка
for i in range(tfidf_matrix.shape[0]):
    for j in range(2):
        kldistances[i, j] = kullback_leibler_divergence(tfidf_matrix[i].toarray().flatten(), cluster_centers[j])

cluster_assignment = np.argmin(kldistances, axis=1)

# Print a few samples from each cluster
for cluster_id in range(2):
    cluster_samples = [text_data[i] for i, label in enumerate(cluster_assignment) if label == cluster_id]
    print(f"Cluster {cluster_id+1} samples:")
    for sample in cluster_samples[:10]:  # Print the first 10 samples
        print(sample)
    print("\n")

from sklearn.decomposition import PCA
# Apply PCA to reduce dimensions
pca = PCA(n_components=2)
tfidf_reduced = pca.fit_transform(tfidf_matrix.toarray())

# Plot the clusters
plt.scatter(tfidf_reduced[:, 0], tfidf_reduced[:, 1], c=cluster_assignment)
plt.title("Cluster Visualization")
plt.show()


