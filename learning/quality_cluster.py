from scipy import stats


def chose(x, clusters):
	if str(x) in clusters:
		return clusters[str(x)]
	return '0'
		
def set_cluster(df, clusters):
	df['#Cluster'] = df['Square'].apply(lambda x: chose(x, clusters))
	
	
def compute_anova(df, clusters):
	clusters = clusters['labels']
	set_cluster(df, clusters)
	
	clusters_ = set(clusters.values())
	# print(stats.wilcoxon(list(df[df['#Cluster'] == '0']['SMSin'])))
	return stats.levene(*[df[df['#Cluster'] == c]['SMSin'] for c in clusters_])
	
	# f_val, p_val = stats.f_oneway(*[df[df['#Cluster'] == c]['SMSin'] for c in clusters_])  
	# print("One-way ANOVA P =", p_val )
	
