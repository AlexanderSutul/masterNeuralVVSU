# coding=utf-8
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

import pickle
import csv


def normalize():
    pass


def denormalize():
    pass


def denormilize_out_param(sample):
    pass


def save_data(net, sample_name):  # сохранение сети в файл
    file_name = '{0}'.format(str(sample_name))
    file_obj = open(file_name, 'w')
    pickle.dump(net, file_obj)
    file_obj.close()
    return "File saved. With name {0}.".format(str(file_name))


def load_data(sample_name):  # загрузка сети в файл
    file_obj = open("{0}".format(str(sample_name)), 'r')
    net = pickle.load(file_obj)
    return net


def get_data(filename):  # получение первичных данных из csv файла
    doc = open(filename, 'rb')
    reader = csv.reader(doc)
    formatted_data = []
    for row in reader:
        for items in row:
            splitted_items = items.split(';')
            customed = [(float(elem)) for elem in splitted_items]
            formatted_data.append(customed)
    ds = SupervisedDataSet(9, 1)
    for row in formatted_data:
        # print(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
        ds.addSample((row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]), (row[9]))
    return ds


def start():
    pass


ds = get_data("data_for_analize_MEP.csv")

net = buildNetwork(9, 3, 1)
trainer = BackpropTrainer(net, ds)

sample_name = "mep"

epochs = 1000
trainer.trainUntilConvergence(maxEpochs=epochs, validationProportion=0.99, verbose=True)

save_data(net, sample_name)
newNet = load_data(sample_name)

result = newNet.activate([19, 22647, 60, 78, 22.86236854, 36, 27, 36, 33820.475])

print(result)
