
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    "Annual_Income": [15,16,17,18,19,40,42,43,44,45,70,72,74,75,78],
    "Spending_Score": [39,40,42,38,40,60,65,63,62,64,20,22,19,18,21]
}

df = pd.DataFrame(data)

X = df[["Annual_Income","Spending_Score"]]

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

df["Cluster"] = kmeans.labels_

print(df)

print(kmeans.cluster_centers_)

plt.scatter(X["Annual_Income"], X["Spending_Score"], c=kmeans.labels_)
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("KMeans Clustering")
plt.show()
