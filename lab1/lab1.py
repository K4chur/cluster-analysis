import random
import matplotlib.pyplot as plt
import numpy as np
from plotly.figure_factory._dendrogram import sch
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix


def calculate_distance(vector1, vector2, method="Евклідова",p = 3, W=None):
    if len(vector1) != len(vector2):
        raise ValueError("Вектори повинні мати однакову довжину")

    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    if method == "Евклідова":
        distance = np.linalg.norm(vector1 - vector2)
    elif method == "Міньковського":
          # Можете змінити потрібне значення p
        distance = np.power(np.sum(np.abs(vector1 - vector2) ** p), 1 / p)
    elif method == "Махалонобіса":
        if W is None:
            raise ValueError("Не вказана матриця коваріації (W) для відстані Махалонобіса")
        if len(vector1) != len(vector2) or len(vector2) != len(W):
            raise ValueError("Вектори та матриця W повинні мати однаковий розмір")
        diff = vector1 - vector2
        diff_transpose = diff.reshape(1, -1)
        distance = np.sqrt(np.dot(np.dot(diff_transpose, W), diff))[0]
    elif method == "Манхеттенська":
        distance = np.sum(np.abs(vector1 - vector2))
    elif method == "Чебишова":
        distance = np.max(np.abs(vector1 - vector2))
    else:
        raise ValueError("Непідтримуваний метод обчислення відстані")

    return distance


# vector1 = [1, 2, 3]
# vector2 = [-2, 4, 3]
# W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Приклад додатньо-визначеної матриці коваріації
#
# method = "Евклідова"
# distance = calculate_distance(vector1, vector2, method)
# print(f"Відстань {method}: {distance}")
#
# method = "Міньковського"
# distance = calculate_distance(vector1, vector2, method)
# print(f"Відстань {method}: {distance}")
#
# method = "Махалонобіса"
# distance = calculate_distance(vector1, vector2, method, W)
# print(f"Відстань {method}: {distance}")

##############################################################################################

def compute_distance_matrix(data, method="Евклідова", W=None):
    N = data.shape[0]
    distance_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            if i == j:
                distance_matrix[i, j] = 0
            else:
                distance = calculate_distance(data[i], data[j], method, W)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    return distance_matrix


def compute_similarity_matrix(data):
    N = data.shape[0]
    similarity_matrix = np.zeros((N, N))
    max_distance = np.max(data)

    for i in range(N):
        for j in range(i, N):
            similarity = 1 - (data[i, j] / max_distance)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    return similarity_matrix


def visualize_matrix(distance_matrix, type):
    plt.imshow(distance_matrix, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label=type)
    plt.title(type + ' Matrix')
    plt.xlabel('Objects')
    plt.ylabel('Objects')
    plt.show()


##############################################################################################

# data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Приклад даних, де N=3, n=3
# method = "Евклідова"
# distance_matrix = compute_distance_matrix(data, method)
# print("Матриця відстаней:")
# print(distance_matrix)
# visualize_matrix(distance_matrix, 'Дистанція')
#
#
# similarity_matrix = compute_similarity_matrix(distance_matrix)
# print("Матриця подібностей:")
# print(similarity_matrix)
# visualize_matrix(similarity_matrix, 'Подібність')

array = []
for i in range(10):
    array.append([random.uniform(-10, 10), random.uniform(-10, 10)])

points = np.array(array)

plt.scatter(points[:, 0], points[:, 1], c='b', marker='o', label='Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Random Points in R^2')
plt.legend()
plt.grid(True)
plt.show()

method = "Евклідова"
distance_matrix = compute_distance_matrix(points, method)
print("Матриця відстаней:")
print(distance_matrix.round(3))
visualize_matrix(distance_matrix, 'Дистанція')

similarity_matrix = compute_similarity_matrix(distance_matrix)
print("Матриця подібностей:")
print(similarity_matrix.round(3))
visualize_matrix(similarity_matrix, 'Подібність')

##############################################################################################
mu1 = np.array([7, 1])
mu2 = np.array([1, -9])
mu3 = np.array([-7, -2])

# Коваріаційні матриці для трьох множин
sigma1 = np.array([[0.7, 1], [1, 2]])
sigma2 = np.array([[1, -1], [-1, 2]])
sigma3 = np.array([[2, 0.25], [0.25, 0.3]])

N = 100

# Генеруємо випадкові дані для трьох множин
data1 = np.random.multivariate_normal(mu1, sigma1, N)
data2 = np.random.multivariate_normal(mu2, sigma2, N)
data3 = np.random.multivariate_normal(mu3, sigma3, N)

plt.scatter(data1[:, 0], data1[:, 1], c='r', marker='o', s=10, label='Множина 1')
plt.scatter(data2[:, 0], data2[:, 1], c='g', marker='o', s=10, label='Множина 2')
plt.scatter(data3[:, 0], data3[:, 1], c='b', marker='o', s=10, label='Множина 3')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.title('Множини нормально розподілених випадкових чисел в R^2')
plt.legend()
plt.grid(True)
plt.show()

data = np.vstack((data1, data2, data3))
k_values = [2, 3, 4]
##############################################################################################

for k in k_values:
    # Кластеризація даних методом k-середніх
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Відображення результатів кластеризації
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=20)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Центри')
    plt.title(f'Кластеризація методом k-середніх (k = {k})')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.legend()
    plt.show()

##############################################################################################

for k in k_values:
    # Кластеризація даних методом ієрархічної кластеризації
    clustering = AgglomerativeClustering(n_clusters=k)
    labels = clustering.fit_predict(data)

    # Відображення результатів кластеризації
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=20)
    plt.title(f'Ієрархічна кластеризація (k = {k})')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.show()

##############################################################################################


# Генерація невеликого набору даних (N = 6)
small_data = np.array([[2, 3],
                       [5, 4],
                       [9, 6],
                       [4, 7],
                       [9, 6],
                       [8, 2]])

# Розрахунок матриці відстаней і ієрархічної кластеризації
linkage_matrix = sch.linkage(small_data, method='ward')

# Побудова дендрограми
plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, orientation='right', labels=[f'Data {i + 1}' for i in range(small_data.shape[0])],
           distance_sort='descending')
plt.title('дендрограма ієрархічної кластеризації')
plt.xlabel('Відстань')
plt.show()

##############################################################################################


k = 3

# Виконайте кластеризацію методом k-середніх
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)
predicted_k_labels = kmeans.labels_  # Мітки кластерів, передбачені алгоритмом

actual_labels = [0] * N + [1] * N + [2] * N  # Ваші фактичні мітки, визначені за допомогою фіктивних значень

# Побудова матриці невідповідності для k-середніх
confusion_k = confusion_matrix(actual_labels, predicted_k_labels)

print("Confusion Matrix for K-means:")
print(confusion_k)

# Виконайте ієрархічну кластеризацію
clustering = AgglomerativeClustering(n_clusters=k)
predicted_hierarchical_labels = clustering.fit_predict(data)  # Мітки кластерів, передбачені алгоритмом

# Побудова матриці невідповідності для ієрархічної кластеризації
confusion_hierarchical = confusion_matrix(actual_labels, predicted_hierarchical_labels)

print("Confusion Matrix for Hierarchical:")
print(confusion_hierarchical)

row_k_ind, col_k_ind = linear_sum_assignment(-confusion_k)  # Додаємо мінус перед матрицею для максимізації
sorted_confusion_k_matrix = confusion_k[row_k_ind][:, col_k_ind]

print("Sorted Confusion Matrix for K-means:")
print(sorted_confusion_k_matrix)

row_hierarchical_ind, col_hierarchical_ind = linear_sum_assignment(
    -confusion_hierarchical)  # Додаємо мінус перед матрицею для максимізації
sorted_confusion_hierarchical_matrix = confusion_hierarchical[row_hierarchical_ind][:, col_hierarchical_ind]

print("Sorted Confusion Matrix for Hierarchical:")
print(sorted_confusion_hierarchical_matrix)

errors_k = np.sum(sorted_confusion_k_matrix) - np.sum(np.diag(sorted_confusion_k_matrix))

# Обчислюємо кількість помилок для ієрархічної кластеризації
errors_hierarchical = np.sum(sorted_confusion_hierarchical_matrix) - np.sum(
    np.diag(sorted_confusion_hierarchical_matrix))

# Обчислюємо загальну кількість точок
total_points_k = np.sum(sorted_confusion_k_matrix)
total_points_hierarchical = np.sum(sorted_confusion_hierarchical_matrix)

# Обчислюємо відсоток помилок для K-means та ієрархічної кластеризації
error_percentage_k = (errors_k / total_points_k) * 100
error_percentage_hierarchical = (errors_hierarchical / total_points_hierarchical) * 100

print(f"\nВідсоток помилок для K-means: {error_percentage_k}%")
print(f"Відсоток помилок для Hierarchical: {error_percentage_hierarchical}%")
