from init_data import init_data, apply_overload
from mining.src.model.mining import *
from learning.model.predict import *
from learning.quality_cluster import *


cluster = 4

df = apply_overload(init_data(10))
clustering = ClusteringResult(5, apply_overload(init_data(10)))
prediction = PredictionResult(df)

kmeans = clustering.get_kmeans_result(cluster)
isola = clustering.get_isolation_forest_result()
dbscan = clustering.get_DBSCAN_result()
ward = clustering.get_hierarchical_result()

tech = {
	'kmeans': ('kmeans', compute_anova(df, kmeans)),
	'isola': ('isolation forest', compute_anova(df, isola)),
	'dbscan': ('dbscan', compute_anova(df, dbscan)),
	'ward': ('ward', compute_anova(df, ward))
}

for i in ['kmeans', 'isola', 'dbscan', 'ward']:
	print(tech[i])
	print('#'*30)
