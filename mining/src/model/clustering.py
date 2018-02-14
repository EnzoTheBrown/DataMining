from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.cluster.hierarchy import ward, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


rng = np.random.RandomState(42)


def apply_kmeans(df, nb_clusters, label=True):
    X = df.ix[:, 1:len(df)-1].values
    labels = df.values
    scaler = StandardScaler()
    normalized = scaler.fit_transform(X)
    pca = PCA(n_components=len(X[0])-1)
    pca.fit(normalized)
    X_pca = pca.fit_transform(normalized)
    kmeansoutput = KMeans(n_clusters=nb_clusters, random_state=0).fit(X_pca)

    if label:
        return kmeansoutput.labels_, X_pca
    return kmeansoutput, X_pca
	
	
def apply_dbscan(df, label=True):
    X = df.ix[:, 1:len(df)-1].values
    labels = df.values
    scaler = StandardScaler()
    normalized = scaler.fit_transform(X)
    pca = PCA(n_components=len(X[0])-1)
    pca.fit(normalized)
    X_pca = pca.fit_transform(normalized)
    db = DBSCAN(eps=.5, min_samples=5).fit(X_pca)
    if label:
        return db.labels_
    return db, X_pca


def apply_isolation_forsest(X):
    clf = IsolationForest(random_state=rng, contamination=0.1)
    clf.fit(X)
    res = clf.predict(X)
    return res

def hierarchichal_ward(X):
   Z = ward(X)
   label=fcluster(Z, 3000, criterion='distance', depth=2, R=None, monocrit=None)
   return label
