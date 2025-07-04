import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Create simple fake data
X = []
for _ in range(20):
    income = random.randint(15, 100)
    score = random.randint(1, 100)
    X.append([income, score])

# Step 2: Run Elbow Method
wcss = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 6), wcss, 'bo-')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.tight_layout()
plt.show()
plt.close()

# Step 3: Apply KMeans with K=3
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X)

# Step 4: Plot Results
x_vals = [point[0] for point in X]
y_vals = [point[1] for point in X]

plt.figure()
plt.scatter(x_vals, y_vals, c=labels, cmap='rainbow', edgecolor='k')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.title('Customer Clustering (Fake Data)')
plt.tight_layout()
plt.show()
plt.close()

# Step 5: Show Silhouette Score
score = silhouette_score(X, labels)
print("Silhouette Score:", score)
