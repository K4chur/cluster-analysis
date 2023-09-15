import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from gk import GK
from gg import GG

mu1 = np.array([1, 1])
mu2 = np.array([1, -9])
mu3 = np.array([-7, -2])

# Коваріаційні матриці для трьох множин
sigma1 = np.array([[1, 1], [1, 2]])
sigma2 = np.array([[1, -1], [-1, 2]])
sigma3 = np.array([[2, 0.5], [0.5, 0.3]])

N = 10

# Генеруємо випадкові дані для трьох множин
data1 = np.random.multivariate_normal(mu1, sigma1, N)
data2 = np.random.multivariate_normal(mu2, sigma2, N)
data3 = np.random.multivariate_normal(mu3, sigma3, N)

actual_labels = [0] * N + [1] * N + [2] * N  # Фактичні мітки

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

c_values = [2, 3, 4]
m = 2
epsilon = 0.005

for c in c_values:
    # Виконуємо нечітку с-кластеризацію
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, c, m, epsilon, maxiter=1000)

    # Матриця ймовірностей μij
    mu = u.T
    print(f"\nМатриця ймовірностей с-середніх з {c} кластерами:")
    print(mu)
    clusters = np.argmax(mu, axis=1)
    print(f"Вектор cl с-середніх з {c} кластерами:")
    print(clusters)
    ##################################################################################################

    gk_clustering = GK(n_clusters=c, max_iter=1000, m=m, error=epsilon)
    centers = gk_clustering.fit(data)
    clastersGK = gk_clustering.predict(data)
    uGK = gk_clustering.last_u
    print(f"\nМатриця ймовірностей GK з {c} кластерами:")
    print(uGK.T)
    print(f"Вектор cl GK з {c} кластерами:")
    print(clastersGK)

    ##################################################################################################

    gg_clustering = GG(n_clusters=c, max_iter=1000, m=m, error=epsilon)
    centers = gk_clustering.fit(data)
    clastersGG = gk_clustering.predict(data)
    uGG = gk_clustering.last_u

    # Матриця ймовірностей GG
    muGG = uGG.T
    print(f"\nМатриця ймовірностей GG з {c} кластерами:")
    print(muGG)

    # Вектор cl GG
    clustersGG = np.argmax(muGG, axis=1)
    print(f"Вектор cl GG з {c} кластерами:")
    print(clustersGG)

    ##################################################################################################

    confusion_c = confusion_matrix(actual_labels, clusters)
    confusion_GK = confusion_matrix(actual_labels, clastersGK)
    confusion_GG = confusion_matrix(actual_labels, clustersGG)

    row_c_ind, col_c_ind = linear_sum_assignment(-confusion_c)
    sorted_confusion_c_matrix = confusion_c[row_c_ind][:, col_c_ind]

    row_GK_ind, col_GK_ind = linear_sum_assignment(-confusion_GK)
    sorted_confusion_GK_matrix = confusion_GK[row_GK_ind][:, col_GK_ind]

    row_GG_ind, col_GG_ind = linear_sum_assignment(-confusion_GG)
    sorted_confusion_GG_matrix = confusion_GG[row_GG_ind][:, col_GG_ind]

    errors_c = np.sum(sorted_confusion_c_matrix) - np.sum(np.diag(sorted_confusion_c_matrix))

    errors_GK = np.sum(sorted_confusion_GK_matrix) - np.sum(
        np.diag(sorted_confusion_GK_matrix))

    errors_GG = np.sum(sorted_confusion_GG_matrix) - np.sum(
        np.diag(sorted_confusion_GG_matrix))

    total_points_c = np.sum(sorted_confusion_c_matrix)
    total_points_GK = np.sum(sorted_confusion_GK_matrix)
    total_points_GG = np.sum(sorted_confusion_GG_matrix)

    # Обчислюємо відсоток помилок для K-means та ієрархічної кластеризації
    error_percentage_c = (errors_c / total_points_c) * 100
    error_percentage_GK = (errors_GK / total_points_GK) * 100
    error_percentage_GG = (errors_GG / total_points_GG) * 100

    print(f"\nВідсоток помилок для C: {error_percentage_c}%")
    print(f"Відсоток помилок для GK: {error_percentage_GK}%")
    print(f"Відсоток помилок для GG: {error_percentage_GG}%")
