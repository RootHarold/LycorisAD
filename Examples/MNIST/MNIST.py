"""An example based on the MNIST dataset.

Author: RootHarold
Project: https://github.com/RootHarold/LycorisAD
Data source: http://odds.cs.stonybrook.edu/mnist-dataset/
"""

from LycorisAD import AnomalyDetection
import scipy.io as sio
import random
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    print(AnomalyDetection.version())

    X = sio.loadmat('mnist.mat')["X"]
    X = (X + 138) / 393.0

    normal = X[:6903]
    random.shuffle(normal)
    normal = normal[:6900]

    anomaly = X[6903:]
    random.shuffle(anomaly)

    data1 = normal[:5560]
    data2 = normal[5560:6210]
    data3 = anomaly[:630]

    test_normal = normal[6210:]
    test_anomaly = anomaly[630:]

    conf = {"capacity": 64, "dimension": 100, "nodes": 300, "connections": 6000, "depths": 4, "batch_size": 8,
            "epoch": 1, "evolution": 1, "verbose": True}
    lad = AnomalyDetection(conf)
    
    lad.encode(data1, data2, data3)

    result1 = lad.detect(test_normal)
    result2 = lad.detect(test_anomaly)

    auc1 = [0] * len(result1) + [1] * len(result2)
    auc2 = []
    for item in (result1 + result2):
        if item[0]:
            auc2.append(0)
        else:
            auc2.append(1)
    print("AUC:", roc_auc_score(auc1, auc2))
