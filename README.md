![logo](https://github.com/RootHarold/LycorisAD/blob/master/logo/logo.svg)

**LycorisAD** is an elegant outlier detection algorithm framework based on *AutoEncoder*.

The neural network is built with [**LycorisNet**](https://github.com/RootHarold/Lycoris), and the threshold of reconstruction error is calculated by genetic algorithm (based on [deap](https://github.com/DEAP/deap)).

# Features
* Lightweight and elegant.
* Need not to manually design the structure of the autoencoder.
* Provide minimalist APIs to make development more efficient.

# Installation
The project is based on LycorisNet, and the installation of LycorisNet can be found [**here**](https://github.com/RootHarold/Lycoris#Installation).

```
pip install LycorisAD
```

# Documents
The APIs provided by **AnomalyDetection** (`from LycorisAD import AnomalyDetection`):

Function | Description |  Inputs | Returns
-|-|-|-
**AnomalyDetection**(config) | Constructor. | **config**: The configuration information, including 12 configuration fields. | An object of the class AnomalyDetection.
**encode**(data, normals, anomalies) | Self-encode the samples and calculate the threshold. | **data**: Normal samples for self-encoding.<br/> **normals**: Normal samples used to calculate the threshold.<br/> **anomalies**: Anomaly samples used to calculate the threshold. |
**detect**(data) | Detect samples. | **data**: Samples to be detected. | The results after detecting the samples are returned as a list. There are two fields, the first is a Boolean value, and the second is the reconstruction error. Where 'True' indicates normal and 'False' indicates anomaly.
**save**(path1, path2) | Save the model and related configurations. | **path1**: The path to store the model.<br/> **path2**: The path to store the configurations. |
`@staticmethod`<br/>**load**(path1, path2) | Import pre-trained models and related configurations. | **path1**: The path to import the model.<br/> **path2**: The path to import the configurations. |
**set_config**(config) | Set the configuration information of AnomalyDetection. | **config**: The configuration information, including 12 configuration fields. |
**set_lr**(learning_rate) | Set the learning rate of the AutoEncoder. | **learning_rate**: The learning rate of the AutoEncoder. | 
**set_workers**(workers) | Set the number of worker threads to train the model. | **workers**: The number of worker threads. | 
**get_threshold**() |  |  | Get the threshold.
`@staticmethod`<br/>**version**() |  |  | Returns the version information of AnomalyDetection.

Configurable fields:

Field | Description |Default
-|-|-
capacity | Capacity of LycorisNet. |
dimension | Dimension of samples. |
nodes | The number of hidden nodes added for each neural network. |
connections| The number of connections added for each neural network. |
depths| Total layers of each neural network. |
batch_size| Batch size. |
epoch| Epoch. |
evolution| Number of LycorisNet evolutions. | 0
population| Population in the genetic algorithm. | 256
generation| Generation in the genetic algorithm. | 16
weight| The weights used to calculate the threshold.<br/>The first parameter determines the prediction accuracy of normal samples, while the second parameter determines that of anomaly samples. | [1, 1]
verbose| Whether to output intermediate information. | False

# Examples
* **MNIST**

For a description of the data and its source, see [here](https://github.com/RootHarold/LycorisAD/tree/master/Examples/MNIST).

Import the dependent modules:

```python
from LycorisAD import AnomalyDetection
import scipy.io as sio
import random
from sklearn.metrics import roc_auc_score
```

Read the data and divide it:

```python
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
```

Prepare the configuration information and instantiate the **AnomalyDetection** object:

```python
conf = {"capacity": 64, "dimension": 100, "nodes": 300, "connections": 6000, "depths": 4, "batch_size": 8,
        "epoch": 1, "evolution": 1, "verbose": True}
lad = AnomalyDetection(conf)
```

Self-encoding:

```python
lad.encode(data1, data2, data3)
```

Calculate the AUC on the test set:

```python
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
```

* *More examples will be released in the future.*

# License
LycorisAD is released under the [LGPL-3.0](https://github.com/RootHarold/LycorisAD/blob/master/LICENSE) license. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.