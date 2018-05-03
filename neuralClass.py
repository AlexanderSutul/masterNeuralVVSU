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
    RESULT = []

    # TODO сделать здесь подключение к бд чтобы записывать туда эти значения

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

    normalized_sexes = []
    normalized_ages = []
    normalized_shoulders = []
    normalized_heights = []
    normalized_chests = []
    normalized_body_index_masses = []
    normalized_body_mass = []
    normalized_leans = []
    normalized_forearms = []
    normalized_shins = []
    normalized_out = []

    def __init__(self, file_name, sample_name, query_type):
        self.file_name = file_name
        self.sample_name = sample_name
        self.query_type = query_type
        self.start()

    def __call__(self, *args, **kwargs):
        print(self.RESULT)

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

    def denormalize(self, values):  # денормализация значений
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
            # data_set.addSample((row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]),
            #                  (row[10]))
            self.sexes.append(row[0])
            self.ages.append(row[1])
            self.shoulders.append(row[2])
            self.heights.append(row[3])
            self.chests.append(row[4])
            self.body_index_masses.append(row[5])
            self.body_mass.append(row[6])
            self.leans.append(row[7])
            self.forearms.append(row[8])
            self.shins.append(row[9])
            self.out.append(row[10])

        self.normalized_sexes = self.normalize(self.sexes)
        self.normalized_ages = self.normalize(self.ages)
        self.normalized_shoulders = self.normalize(self.shoulders)
        self.normalized_heights = self.normalize(self.heights)
        self.normalized_chests = self.normalize(self.chests)
        self.normalized_body_index_masses = self.normalize(self.body_index_masses)
        self.normalized_body_mass = self.normalize(self.body_mass)
        self.normalized_leans = self.normalize(self.leans)
        self.normalized_forearms = self.normalize(self.forearms)
        self.normalized_shins = self.normalize(self.shins)
        self.normalized_out = self.normalize(self.out)
        print(self.normalized_out)

        for sex, age, shoulder, height, chest, bim, bm, lean, forearm, shin, output in zip(
                self.normalized_sexes['out']['data'], self.normalized_ages['out']['data'],
                self.normalized_shoulders['out']['data'], self.normalized_heights['out']['data'],
                self.normalized_chests['out']['data'],
                self.normalized_body_index_masses['out']['data'],
                self.normalized_body_mass['out']['data'], self.normalized_leans['out']['data'],
                self.normalized_forearms['out']['data'], self.normalized_shins['out']['data'],
                self.normalized_out['out']['data']):
            print(sex, age, shoulder, height, chest, bim, bm, lean, forearm, shin, output)
            data_set.addSample((sex, age, shoulder, height, chest, bim, bm, lean, forearm, shin), (output))

        return data_set

    def train_net(self, data_set, epochs, validProp, verbose):  # тренировка данных
        net = buildNetwork(10, 3, 1)
        trainer = BackpropTrainer(net, data_set)
        trainer.trainUntilConvergence(maxEpochs=epochs, validationProportion=validProp, verbose=verbose)
        return net

    def get_result(self, net):
        print("self.activation_data")
        input_data1 = [0.0, 0.3287671232876712, 0.25789786664684455, 0.47761194029850745, 0.5882352941176471,
                       0.5318417604869441, 0.3, 0.4117647058823529, 0.45, 0.3453618706781417]  # 1.41939898
        input_data2 = [1.0, 0.1780821917808219, 1.0, 0.5223880597014925, 0.29411764705882354, 0.2248854027747037, 0.35, 0.35294117647058826, 0.4, 0.40165179717288246]  # 0.18846153846153846
        input_data3 = [1.0, 0.4246575342465753, 0.6787333680220026, 0.8805970149253731, 0.7843137254901961,
                       0.4799829210225954, 0.65, 0.7647058823529411, 0.5, 0.8341937382791614]  # 0.6307692307692307
        input_data4 = [0.0, 0.3246575342465753, 0.6787333680220026, 0.8805970149253731, 0.7843137254901961,
                       0.4799829210225954, 0.65, 0.7647058823529411, 0.5, 0.8341937382791614]  # 0.6307692307692307
        self.RESULT.append(net.activate(input_data1))
        self.RESULT.append(net.activate(input_data2))
        self.RESULT.append(net.activate(input_data3))
        self.RESULT.append(net.activate(input_data4))
        return self.RESULT

    def start(self):  # запуск приложения
        # if self.query_type == 'get_data_from_csv_file_and_train':
        # Получение данных из csv файла
        # data_set = self.get_data(self.file_name)
        # Тренировка на данных и получение тренированной сети
        # net = self.train_net(data_set, self.EPOCHS, self.VALID_PROP, self.VERBOSE)
        # Сохранение нейросети в файл
        # self.save_data(net, self.sample_name)
        # elif self.query_type == 'get_answer':
        # Загрузка нейросети из файла
        net = self.load_data(self.sample_name)
        # Активация и получение результата
        self.RESULT = self.get_result(net)
