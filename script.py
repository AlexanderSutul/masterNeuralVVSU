# coding=utf-8
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

import pickle
import csv


def normalize(values):  # нормализация значений
    x_min = min(values)
    x_max = max(values)
    b = 1
    a = 0
    result_array = []
    result = {'out': {'min': x_min, 'max': x_max, 'data': result_array}}
    for i in range(len(values)):
        x = values[i]
        y = ((x - x_min) / (x_max - x_min)) * (b - a) + a
        result_array.append(y)
    result['out']['data'] = result_array
    return result


def denormalize(values):  # денормализация значений
    x_min = values['out']['min']
    x_max = values['out']['max']
    data = values['out']['data']
    result_array = []
    for i in range(len(data)):
        x = data[i]
        y = (x_max - x_min) * x + x_min
        result_array.append(y)
    return result_array


def save_data(net, sample_name):  # сохранение сети в файл
    file_name = '{0}'.format(str(sample_name))
    file_obj = open(file_name, 'w')
    pickle.dump(net, file_obj)
    file_obj.close()
    return "File saved with name '{0}'.".format(str(file_name))


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
    ages = []
    for row in formatted_data:
        ds.addSample((row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]), (row[9]))
        ages.append(row[0])
    newages = normalize(ages)
    print(denormalize(newages))
    return ds


def start():
    ds = get_data("data_for_analize_MEP.csv")

    net = buildNetwork(9, 3, 1)
    trainer = BackpropTrainer(net, ds)

    sample_name = "mep"

    epochs = 1000
    trainer.trainUntilConvergence(maxEpochs=epochs, validationProportion=0.99, verbose=True)

    print(save_data(net, sample_name))
    newNet = load_data(sample_name)

    result = newNet.activate([19, 22647, 60, 78, 22.86236854, 36, 27, 36, 33820.475])

    return result


print(start())
