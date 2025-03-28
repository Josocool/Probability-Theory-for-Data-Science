import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
# read CSV ﬁle 
df = pd.read_csv('height_weight.csv')  #  path 
# Extract only height and weight
X = df[['height', 'weight']].values 
# KMeans Clustering 
kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster'] = kmeans.ﬁt_predict(X) 
# Visualization
plt.scatter(df['height'], df['weight'], c=df['cluster'], cmap='rainbow', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='black', marker='X', label='Centers') 
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')
plt.title('CSV-based key/weight clustering')
plt.legend()
plt.grid(True)
plt.show()