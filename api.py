from flask import Flask, render_template
from flask_restful import Resource, Api
from flask_bootstrap import Bootstrap

import json

from mining.src.model.mining import *
from learning.model.predict import *

from init_data import init_data, apply_overload

app = Flask(__name__)
# Bootstrap(app)

api = Api(app)


df = apply_overload(init_data(10))
clustering = ClusteringResult(5, apply_overload(init_data(10)))
prediction = PredictionResult(df)


class Clustering(Resource):
    def get(self, algo, cluster=4):
        d = {}
        if algo == 'apriori':
            # caution wet floor
            d = {}
            pass
            # d = clustering.apply_apriori('SMSin')
        if algo == 'kmeans':
            d = clustering.get_kmeans_result(cluster)
        if algo == 'tree':
            d = clustering.get_isolation_forest_result()
        if algo == 'dbscan':
            d = clustering.get_DBSCAN_result()
        if algo == 'ward':
            d = clustering.get_hierarchical_result()
        
        # preparing the cluster for machine learning
        clusters = prediction.dict2clusters(d['labels'])
        prediction.reset_clusters()
        for key, cluster in clusters.items():
            prediction.get_cluster(key, cluster)
        data = json.dumps(d)
        return data
    

class Prediction(Resource):
    def get(self, dummy):
        d = prediction.predict('0')
        return json.dumps(d)

@app.route('/')
def show_basic():
    return render_template("index.html")


@app.route('/rapport')
def show_rapport():
    return render_template("rapport.html")


@app.route('/predict')
def show_prediction():
    return render_template("index.html")



api.add_resource(Clustering, '/<string:algo>', '/<string:algo>/<int:cluster>')
api.add_resource(Prediction, '/predict/<string:dummy>')

def run():
    app.run(debug=True)


