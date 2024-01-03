import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.stats import entropy

# Завантаження та підготовка даних
data = pd.read_csv('test.csv')
documents = data['text'].tolist()

# Векторизація текстів за допомогою TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Налаштуйте параметри
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Розрахунок відстані Кульбака-Лейблера
def kl_divergence(p, q):
    return entropy(p, q)

# Кластеризація
num_clusters = 2  # Кількість кластерів
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(tfidf_matrix)

# Додайте мітку кластера до даних
data['cluster'] = kmeans.labels_

# Виведення результатів
for cluster_id in range(num_clusters):
    cluster_data = data[data['cluster'] == cluster_id]
    print(f'Cluster {cluster_id}:')
    print(cluster_data['text'].head())
