import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from sklearn.cluster import Birch, KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from pyclustering.cluster.cure import cure
from pyclustering.cluster import cluster_visualizer


def hopkins_statistic(data, n_neighbors=10):
    """
    Calculate the Hopkins statistic to assess clustering tendency.

    Parameters:
    - data: The dataset for which clustering tendency is to be assessed.
    - n_neighbors: Number of nearest neighbors used for calculation.

    Returns:
    - Hopkins statistic value.
    """
    n = len(data)
    rand_data = np.random.rand(n, data.shape[1])

    # Calculate the distances between data points and random points
    nn_data = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    nn_rand = NearestNeighbors(n_neighbors=n_neighbors).fit(rand_data)

    u_data = nn_data.kneighbors(data, n_neighbors, return_distance=True)[0]
    u_rand = nn_rand.kneighbors(rand_data, n_neighbors, return_distance=True)[0]

    # Calculate the Hopkins statistic
    h = np.sum(u_data) / (np.sum(u_data) + np.sum(u_rand))
    return h


# Seed for reproducibility
np.random.seed(0)

# Cluster 1
N1 = 10000
mean1 = np.array([-5, -7])
cov1 = np.array([[4, -2], [-2, 2]])
cluster1 = np.random.multivariate_normal(mean1, cov1, N1)

# cluster2_x = np.random.uniform(1, 11, N2)
# cluster2_y = np.random.uniform(1, 11, N2)
# cluster2_mask = (2 * (cluster2_x - 6) ** 2 + 3 * (cluster2_y - 4) ** 2 <= 10)
# cluster2 = np.column_stack((cluster2_x[cluster2_mask], cluster2_y[cluster2_mask]))
# Cluster 2 (Ellipse)
N2 = 15000
cluster2 = []
while len(cluster2) < N2:
    x = np.random.uniform(1, 11)
    y = np.random.uniform(1, 11)
    if 2 * (x - 6) ** 2 + 3 * (y - 4) ** 2 <= 10:
        cluster2.append([x, y])

cluster2 = np.array(cluster2)

# Cluster 3 (Rectangle)
N3 = 10000
cluster3 = []
while len(cluster3) < N3:
    x = np.random.uniform(-12, 5)  # Adjust the range as needed
    y = np.random.uniform(-1, 12)  # Adjust the range as needed

    # Check if the point satisfies the condition
    if (np.abs(x + y) <= 2) and (np.abs(x - y + 15) <= 3):
        cluster3.append([x, y])

cluster3 = np.array(cluster3)

# Outliers
N4 = 500
outliers_x = np.random.uniform(-20, 20, N4)
outliers_y = np.random.uniform(-20, 20, N4)
outliers = np.column_stack((outliers_x, outliers_y))

# Combine all clusters
data = np.vstack((cluster1, cluster2, cluster3, outliers))

# Plot the clusters
plt.figure(figsize=(10, 8))
plt.scatter(cluster1[:, 0], cluster1[:, 1], c="red", label="cluster1", alpha=0.5)
plt.scatter(cluster2[:, 0], cluster2[:, 1], c="blue", label="cluster2", alpha=0.5)
plt.scatter(cluster3[:, 0], cluster3[:, 1], c="purple", label="cluster3", alpha=0.5)
plt.scatter(outliers[:, 0], outliers[:, 1], c="black", label="outliers", alpha=0.5)
plt.title('Generated Data Clusters')
plt.legend(['1stCluster', '2ndCluster', '3rdCluster', 'Outlines'], loc=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

# Save the generated data to a file
np.save('generated_data.npy', data)
###########################################################################################

hopkins_value = hopkins_statistic(data)
print(f"Hopkins Statistic: {hopkins_value:.5f}")

###########################################################################################

# Parameters for bootstrapping
m = int(0.07 * len(data))  # 7% of the data points
B = 10_000  # Number of bootstrap samples

# Initialize an array to store the bootstrapped Hopkins statistics
# bootstrap_hopkins = np.zeros(B)
#
# # Perform bootstrapping
# for i in range(B):
#     # Sample with replacement from the original data
#     bootstrap_sample = data[np.random.choice(len(data), size=m, replace=True)]
#
#     # Calculate the Hopkins statistic for the bootstrap sample
#     bootstrap_hopkins[i] = hopkins_statistic(bootstrap_sample)

# Plot the distribution of bootstrapped Hopkins statistics
# np.save("bootstrap_hopkins.np", bootstrap_hopkins)
bootstrap_hopkins = np.load("bootstrap_hopkins.np.npy")
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_hopkins, bins=30, density=True, alpha=0.5, color='b', label='Bootstrapped Hopkins')
plt.title('Distribution of Bootstrapped Hopkins Statistics')
plt.xlabel('Hopkins Statistic')
plt.ylabel('Density')
plt.grid(True)

# Fit a beta distribution to the bootstrapped data
# params = beta.fit(bootstrap_hopkins)
x = np.linspace(0, 1, 100)
# pdf = beta.pdf(x, *params)
# np.save("pdf.np", pdf)
pdf = np.load("pdf.np.npy")
plt.plot(x, pdf, 'r-', lw=2, label='Beta Distribution')
plt.xlim(0.85,0.95)
plt.legend()
plt.show()

###########################################################################################

# Varying values of NMIN
NMIN = 100  # Adjust as needed

# Keep h constant
h = 0.5  # Adjust as needed

# Create a DBSCAN instance with the specified parameters
dbscan = DBSCAN(eps=h, min_samples=NMIN)

# Fit the DBSCAN model to your data
clusters = dbscan.fit_predict(data)

# Count the number of clusters (excluding outliers, labeled as -1)
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

# Print the number of clusters and the corresponding NMIN value
print(f"NMIN = {NMIN}, Number of Clusters: {num_clusters}")

# Create a scatter plot to visualize the clusters
plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering Results')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

###########################################################################################

k = 3
N = len(data)
alpha = 0.3
alphaN = int(alpha * N)
m = 3
# 431 422 399 366 326
DS = [{'Ni': 0, 'S1i': np.zeros(2), 'S2i': np.zeros(2)} for _ in range(k)]
CS = []  # Initialize CS clusters
RS = []

initial_centers = data[np.random.choice(len(data), k, replace=False)]

# Make the centers the DS at the start
for i in range(k):
    DS[i]['Ni'] += 1
    DS[i]['S1i'] += initial_centers[i]
    DS[i]['S2i'] += initial_centers[i] ** 2


def plot_clusters_and_rs(DS, RS):
    plt.figure(figsize=(8, 6))

    colors = ['blue', 'green', 'red']  # Add more colors if needed

    # Plot DS clusters as ellipses with different colors
    for i, cluster in enumerate(DS):
        if cluster['Ni'] > 1:
            Mi = cluster['S1i'] / cluster['Ni']
            Di = (cluster['S2i'] / (cluster['Ni'] - 1)) - ((cluster['S1i'] / cluster['Ni']) ** 2)
            width = 2 * m * np.sqrt(Di[0])
            height = 2 * m * np.sqrt(Di[1])
            ellipse = Ellipse(Mi, width, height, fill=True, color=colors[i], alpha=0.5, label=f"DSCluster-{i}")
            plt.gca().add_patch(ellipse)

    # Plot RS points
    if np.any(RS):
        rs_data = np.array(RS)
        plt.scatter(rs_data[:, 0], rs_data[:, 1], c='black', marker='x', label='Retained Set')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('DS Clusters and RS Points')
    plt.legend()
    plt.show()


# Main loop: Assign points to DS clusters
data_copy = np.array(data)
while data_copy.shape[0] > 0:
    tempData = data_copy[:alphaN, :]
    data_copy = data_copy[alphaN:, :]

    for point in tempData:
        closest_cluster = None
        closest_distance = float('inf')

        # Find the closest DS cluster for the point
        for i, cluster in enumerate(DS):
            Mi = cluster['S1i'] / cluster['Ni']
            Di = (cluster['S2i'] / (cluster['Ni'] - 1)) - ((cluster['S1i'] / cluster['Ni']) ** 2)
            condition = np.all(Mi - m * np.sqrt(Di) <= point) and np.all(point <= Mi + m * np.sqrt(Di))

            if condition:
                closest_cluster = i

        if closest_cluster is not None:
            # Update statistics
            DS[closest_cluster]['Ni'] += 1
            DS[closest_cluster]['S1i'] += point
            DS[closest_cluster]['S2i'] += point ** 2
        else:
            # Add the point to the Retained Set (RS)
            RS.append(point.tolist())


plot_clusters_and_rs(DS, RS)
# Print statistics of DS clusters
for i, cluster in enumerate(DS):
    print(f'DS Cluster {i + 1}: N={cluster["Ni"]}, S1={cluster["S1i"]}, S2={cluster["S2i"]}')
# Print the size of the RS
print(f'Retained Set (RS) Size: {len(RS)}')


# epsilon = 2  # Adjust this value based on your data
# min_samples = 7  # Adjust this value based on your data
#
# # Convert RS to a NumPy array
# rs_data = np.array(RS)
#
# # Create an instance of DBSCAN
# dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
#
# # Fit DBSCAN to the RS data
# dbscan.fit(rs_data)
#
# # Get cluster labels (-1 represents outliers)
# cluster_labels = dbscan.labels_
#
# # Separate clustered points from outliers
# clustered_rs = rs_data[cluster_labels != -1]
# outliers = rs_data[cluster_labels == -1]
#
# plot_clusters_and_rs(DS, None, outliers)
#
# # Plot clustered points
# plt.scatter(clustered_rs[:, 0], clustered_rs[:, 1], c=cluster_labels[cluster_labels != -1], cmap='viridis', marker='o', label='Clustered RS Points')
#
# # Plot outliers
# plt.scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x', label='Outliers')
#
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('DBSCAN Clustering of RS Points')
# plt.legend()
# plt.show()

M = [3, 3.5, 4, 4.5, 5]
N = [431, 422, 399, 366, 326]

# Побудова графіка
plt.plot(M, N, marker='o', linestyle='-')

# Підписи до вісей і заголовок
plt.xlabel('Значення M')
plt.ylabel('Значення N')
plt.title('Залежність M від N')
plt.show()
# 6

# Parameters
s = 2000  # sample size for hierarchical clustering
p = 4  # number of representative points
q = 5  # final number of clusters
c = 30  # number of clusters for hierarchical clustering
alpha = 0.6  # compression level for representative points

# 2. Extract a sample for hierarchical clustering
sample_indices = np.random.choice(data.shape[0], s, replace=False)
sample_data = data[sample_indices]

# 3. Create CURE object and perform clustering
cure_instance = cure(sample_data.tolist(), q, p, alpha)
cure_instance.process()
clusters = cure_instance.get_clusters()

# 4. Visualize the results
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, sample_data)
visualizer.show()

# 7
L = 100  # branching_factor
nb = 3  # n_clusters
T = 0.3  # threshold
k = 3

# Cluster using BIRCH
birch = Birch(n_clusters=nb, threshold=T, branching_factor=L)
birch_clusters = birch.fit_predict(data)

# Plot the clustered data
plt.figure(figsize=(10, 8))
for i in range(nb):
    plt.scatter(data[birch_clusters == i][:, 0], data[birch_clusters == i][:, 1], s=1, label=f"Cluster {i + 1}")
plt.title("Data Clustered Using BIRCH with Given Parameters")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
