from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris = datasets.load_iris()
X = iris.data

model = KMeans(n_clusters=3, random_state=42)

labels = model.fit_predict(X)

score = silhouette_score(X, labels)
print("Silhouette Score:", score)
