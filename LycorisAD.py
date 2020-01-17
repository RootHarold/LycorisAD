from LycorisNet import Lycoris
from LycorisNet import loadModel
from deap import creator, base, tools, algorithms
from scipy.stats import bernoulli
import math
import random
import numpy as np
import json
import logging

'''
    Copyright (c) 2020, RootHarold
    All rights reserved.
    Use of this source code is governed by a LGPL-3.0 license that can be found
    in the LICENSE file.
'''

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)


class AnomalyDetection:

    def __init__(self, config):
        if config is not None:
            self.__check_config(config)
            self.__config = config
            self.__lie = Lycoris(capacity=config["capacity"], inputDim=config["dimension"],
                                 outputDim=config["dimension"], mode="predict")
            self.__lie.setMutateOdds(0)
            self.__lie.preheat(config["nodes"], config["connections"], config["depths"])
            self.__ret_pos = []
            self.__ret_neg = []
            self.__threshold = 0.0
            self.__max_num = 0.0
            self.__flag = True

    def encode(self, data, normals, anomalies):
        if np.array(data).ndim == 1:
            data = [data]

        if np.array(normals).ndim == 1:
            normals = [normals]

        if np.array(anomalies).ndim == 1:
            anomalies = [anomalies]

        flag = True
        if self.__flag:
            self.__flag = False
        else:
            flag = False

        data_copy = data.copy()
        batch = math.ceil(len(data) / float(self.__config["batch_size"]))
        remainder = len(data) % self.__config["batch_size"]

        for i in range(self.__config["epoch"]):
            for j in range(remainder):
                data_copy.append(random.choice(data))

            random.shuffle(data_copy)
            temp = [None] * self.__config["batch_size"]
            pos = 0

            for j in range(batch):
                for k in range(self.__config["batch_size"]):
                    temp[k] = data_copy[pos]
                    pos = pos + 1

                if flag:
                    if i * batch + j == self.__config["evolution"]:
                        self.__lie.enrich()

                    if i * batch + j < self.__config["evolution"]:
                        self.__lie.fitAll(temp, temp)
                    else:
                        self.__lie.fit(temp, temp)
                else:
                    self.__lie.fit(temp, temp)

            if self.__config["verbose"]:
                logging.info("Epoch " + str(i + 1) + " : " + str(self.__lie.getLoss()))

        self.__ret_pos.clear()
        normals_ = self.__lie.computeBatch(normals)
        for item, item_ in zip(normals, normals_):
            self.__ret_pos.append(np.linalg.norm(np.array(item) - np.array(item_)))

        self.__ret_neg.clear()
        anomalies_ = self.__lie.computeBatch(anomalies)
        for item, item_ in zip(anomalies, anomalies_):
            self.__ret_neg.append(np.linalg.norm(np.array(item) - np.array(item_)))

        self.__max_num = max(self.__ret_pos) if max(self.__ret_pos) > max(self.__ret_neg) else max(self.__ret_neg)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        GENE_LENGTH = 26

        toolbox = base.Toolbox()
        toolbox.register("binary", bernoulli.rvs, 0.5)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n=GENE_LENGTH)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        pop = toolbox.population(self.__config["population"])

        toolbox.register("evaluate", self.__eval)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        resultPop, _ = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=self.__config["generation"],
                                           stats=stats, verbose=False)
        index = np.argmax([ind.fitness for ind in pop])
        self.__threshold = self.__decode(resultPop[index])

        if self.__config["verbose"]:
            logging.info("Find the threshold: " + str(self.__threshold))

    def detect(self, data):
        if np.array(data).ndim == 1:
            data = [data]

        ret = []
        data_ = self.__lie.computeBatch(data)
        for item, item_ in zip(data, data_):
            error = np.linalg.norm(np.array(item) - np.array(item_))
            ret.append([error < self.__threshold, error])

        return ret

    def save(self, path1, path2):
        self.__lie.saveModel(path=path1)
        config_copy = self.__config.copy()
        config_copy["threshold"] = self.__threshold
        json_info = json.dumps(config_copy, indent=4)
        f = open(path2, 'w')
        f.write(json_info)
        f.close()

        if self.__config["verbose"]:
            logging.info("Model saved successfully.")

    @staticmethod
    def load(path1, path2):
        l_ad = AnomalyDetection(None)
        l_ad.__ret_pos = []
        l_ad.__ret_neg = []
        l_ad.__max_num = 0.0
        l_ad.__flag = False

        l_ad.__lie = loadModel(path1, capacity=1)

        f = open(path2, 'r')
        json_info = f.read()
        f.close()

        config = json.loads(json_info)
        l_ad.__threshold = config["threshold"]
        config.pop("threshold")
        config["capacity"] = 1
        config["evolution"] = 0
        l_ad.__check_config(config)
        l_ad.__config = config
        if l_ad.__config["verbose"]:
            logging.info("Model imported successfully.")

        return l_ad

    def set_config(self, config):
        self.__check_config(config)
        self.__config = config

    def set_lr(self, learning_rate):
        self.__lie.setLR(learning_rate)

    def set_workers(self, workers):
        self.__lie.setCpuCores(num=workers)

    def get_threshold(self):
        return self.__threshold

    @staticmethod
    def version():
        lycoris_version = Lycoris.version()
        return "LycorisAD 1.0-Dev By RootHarold." + "\nPowered By " + lycoris_version[:-15] + "."

    def __decode(self, individual):
        num = int(''.join([str(_) for _ in individual]), 2)
        x = num * math.ceil(self.__max_num) / (2 ** 26 - 1)
        return x

    def __eval(self, individual):
        value = self.__decode(individual)
        fitness = 0.0

        for item in self.__ret_pos:
            if value > item:
                fitness = fitness + self.__config["weight"][0]

        for item in self.__ret_neg:
            if value < item:
                fitness = fitness + self.__config["weight"][1]

        return fitness,

    @staticmethod
    def __check_config(config):
        keys = ["capacity", "dimension", "nodes", "connections", "depths", "batch_size", "epoch"]
        for item in keys:
            if item not in config:
                raise Exception("Invalid configuration.")

        if "evolution" not in config:
            config["evolution"] = 0

        if "population" not in config:
            config["population"] = 256

        if "generation" not in config:
            config["generation"] = 16

        if "weight" not in config:
            config["weight"] = [1, 1]

        if "verbose" not in config:
            config["verbose"] = False
