# coding=utf-8
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

import csv


class NeuralNetMaster:
    EPOCHS = 1000
    VALID_PROP = 0.99
    VERBOSE = True
    RESULT = None

    output_max = 299
    output_min = 39

    sexes = []
    ages = []
    shoulders = []
    heights = []
    chests = []
    body_index_masses = []
    body_mass = []
    leans = []
    forearms = []
    shins = []
    out = []

    def __init__(self, file_name, sample_name, query_type, activation_data):
        self.file_name = file_name
        self.sample_name = sample_name
        self.query_type = query_type
        self.activation_data = activation_data
        self.start()

    def normalize(self, values):  # нормализация значений
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

    def denormalize(self, values, getting_result=False):  # денормализация значений
        if getting_result:
            x_min = self.output_min
            x_max = self.output_max
        else:
            x_min = values['out']['min']
            x_max = values['out']['max']
        data = values['out']['data']
        result_array = []
        for i in range(len(data)):
            x = data[i]
            y = (x_max - x_min) * x + x_min
            result_array.append(y)
        return result_array

    def save_data(self, net, sample_name):  # сохранение сети в файл
        file_name = '{0}'.format(str(sample_name))
        NetworkWriter.writeToFile(net, file_name + '.xml')
        print "File saved with name '{0}'.".format(str(file_name))

    def load_data(self, sample_name):  # загрузка сети в файл
        net = NetworkReader.readFrom(sample_name + '.xml')
        return net

    def get_data(self, filename):  # получение первичных данных из csv файла
        doc = open(filename, 'rb')
        reader = csv.reader(doc)
        formatted_data = []
        for row in reader:
            for items in row:
                splitted_items = items.split(';')
                customed = [(float(elem)) for elem in splitted_items]
                formatted_data.append(customed)
        data_set = SupervisedDataSet(10, 1)
        sexes = []
        ages = []
        shoulders = []
        heights = []
        chests = []
        body_index_masses = []
        body_mass = []
        leans = []
        forearms = []
        shins = []
        out = []
        output_normalized = []

        for row in formatted_data:
            data_set.addSample((row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]),
                               (row[10]))

        return data_set

    def train_net(self, data_set, epochs, validProp, verbose):  # тренировка данных
        net = buildNetwork(10, 5, 1)
        trainer = BackpropTrainer(net, data_set)
        trainer.trainUntilConvergence(maxEpochs=epochs, validationProportion=validProp, verbose=verbose)
        return net

    def get_result(self, net):
        print("self.activation_data")
        print(self.activation_data)
        self.RESULT = net.activate(self.activation_data)
        return self.RESULT

    def start(self):  # запуск приложения

        if self.query_type == 'get_data_from_csv_file_and_train':
            # Получение данных из csv файла
            data_set = self.get_data(self.file_name)
            # Тренировка на данных и получение тренированной сети
            net = self.train_net(data_set, self.EPOCHS, self.VALID_PROP, self.VERBOSE)
            # Сохранение нейросети в файл
            self.save_data(net, self.sample_name)
        elif self.query_type == 'get_answer':
            # Загрузка нейросети из файла
            net = self.load_data(self.sample_name)
            # Активация и получение результата
            result = self.get_result(net)
            print(result)


# Начало работы программы
# input_data = [1, 19, 31048, 70, 85, 20.45288532, 37, 26, 34, 39714.94834]  # 132.87976709
# input_data = [1, 29, 24108, 52, 70, 18.87066338, 28, 21, 36, 27303.47385]  # 132.87976709
input_data = [0, 48, 23377, 80, 90, 29.74419988, 35, 29, 33, 36685.23509]  # 87.30399729
# Запросы:
# get_data_from_csv_file_and_train => Получение данных из файла CSV и тренировка данных и Сохранение
# тренированной сети в файле
# get_answer => Загрузка тренированной сети из файла и получение ответа
query_type = 'get_data_from_csv_file_and_train'

app = NeuralNetMaster("mep_test_data_last.csv", 'mep', query_type, input_data)
