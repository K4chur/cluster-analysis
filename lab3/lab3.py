import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist

# Load the MNIST dataset for digits 0-9
(X_train, y_train), (_, _) = mnist.load_data()

# Flatten the images and normalize pixel values
X_train = X_train.reshape(X_train.shape[0], -1).astype('float32')
X_train /= 255.0

# Labels for digits 0-9
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create a mask to filter only the specified digits
digit_mask = np.isin(y_train, digits)
X_filtered = X_train[digit_mask]
y_filtered = y_train[digit_mask]

# Save the filtered data and labels
np.save('X_filtered.npy', X_filtered)
np.save('y_filtered.npy', y_filtered)

# Load the filtered data
X_filtered = np.load('X_filtered.npy')
y_filtered = np.load('y_filtered.npy')

cov_matrix = np.cov(X_filtered.T)
eigenvalues, _ = np.linalg.eig(cov_matrix)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='b')
plt.title('Власні значення коваріаційної матриці')
plt.xlabel('Номер власного значення')
plt.ylabel('Власні значення')
plt.grid(True)
plt.show()

# Find the indices of nonzero pixels
nonzero_columns = np.any(X_filtered != 0, axis=0)

# Filter the data based on nonzero columns
filtered_data = X_filtered[:, nonzero_columns]

# Create a new covariance matrix using the filtered data
cov_matrix = np.cov(filtered_data.T)

# Calculate the eigenvalues and eigenvectors
eigenvalues, _ = np.linalg.eig(cov_matrix)

# Sort the eigenvalues in descending order
sorted_eigenvalues = np.sort(eigenvalues)[::-1]

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, marker='o', linestyle='-', color='b')
plt.title('Eigenvalues of the covariance matrix')
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()

# Calculate the cumulative explained variance
cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.title('Explained Variance vs. Number of Pixels')
plt.xlabel('Number of Pixels')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.show()

# Find the number of eigenvalues that explain at least 95% of the variance
N = np.argmax(cumulative_variance >= 0.95) + 1

print(f"Number of pixels to retain 95% variance: {N}")

# Reduce dimensionality to N components
X_reduced = X_filtered[:, :N]

# Perform K-Means clustering with k = 10 on reduced data
kmeans = KMeans(n_clusters=10, random_state=0)
cluster_labels = kmeans.fit_predict(X_reduced)
np.save("cluster_labels.npy", cluster_labels)

# Calculate the confusion matrix for reduced data
conf_matrix = confusion_matrix(y_filtered, cluster_labels)

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

row_k_ind, col_k_ind = linear_sum_assignment(-conf_matrix)  # Додаємо мінус перед матрицею для максимізації
conf_matrix = conf_matrix[row_k_ind][:, col_k_ind]
print("Sorted conf matrix:")
print(conf_matrix)

# Calculate the total number of samples for reduced data
total_samples = np.sum(conf_matrix)

# Calculate the sum of the diagonal elements (correctly classified samples)
correctly_classified = np.sum(np.diag(conf_matrix))

# Calculate the sum of all elements except the diagonal (misclassified samples)
misclassified = total_samples - correctly_classified

# Calculate the percentage of errors for reduced data
error_percentage = (misclassified / total_samples) * 100

print(f"Percentage of errors with reduced data: {error_percentage:.2f}%")

# Perform K-Means clustering with k = 10 on full data
kmeansX = KMeans(n_clusters=10, random_state=0)
cluster_labelsX = kmeansX.fit_predict(X_filtered)
np.save("cluster_labelsX.npy", cluster_labelsX)

# Calculate the confusion matrix for full data
conf_matrixX = confusion_matrix(y_filtered, cluster_labelsX)

# Display the confusion matrix
print("Confusion Matrix (Full Data):")
print(conf_matrixX)

row_kX_ind, col_kX_ind = linear_sum_assignment(-conf_matrixX)  # Додаємо мінус перед матрицею для максимізації
conf_matrixX = conf_matrixX[row_kX_ind][:, col_kX_ind]
print("Sorted conf matrix:")
print(conf_matrixX)

# Calculate the total number of samples for full data
total_samplesX = np.sum(conf_matrixX)

# Calculate the sum of the diagonal elements (correctly classified samples)
correctly_classifiedX = np.sum(np.diag(conf_matrixX))

# Calculate the sum of all elements except the diagonal (misclassified samples)
misclassifiedX = total_samplesX - correctly_classifiedX

# Calculate the percentage of errors for full data
error_percentageX = (misclassifiedX / total_samplesX) * 100

print(f"Percentage of errors with full data: {error_percentageX:.2f}%")
