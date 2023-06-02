from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans

def agglomerative_clustering(embeddings, n_clusters=4):
    agglo = AgglomerativeClustering(n_clusters=n_clusters) 
    agglo.fit(embeddings)
    labels = agglo.labels_
    return labels

def dbscan(embeddings, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples) 
    dbscan.fit(embeddings)
    labels = dbscan.labels_
    return labels

def kmeans(embeddings, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters) 
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels