#!/usr/bin/python
# coding=utf-8
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, TanhLayer, FullConnection

from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

import csv

from db import *
from additionalFunctions import *

class NeuralNetMaster:
    EPOCHS = 10000  # TODO: переставить потом на 5000
    RESULT = []

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
    mep = []
    mip = []
    snip = []

    sexes_test = []
    ages_test = []
    shoulders_test = []
    heights_test = []
    chests_test = []
    body_index_masses_test = []
    body_mass_test = []
    leans_test = []
    forearms_test = []
    shins_test = []
    mep_test = []
    mip_test = []
    snip_test = []

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
    normalized_mep = []
    normalized_mip = []
    normalized_snip = []

    def __init__(self, file_name, sample_name, query_type, data):
        self.file_name = file_name
        self.sample_name = sample_name
        self.query_type = query_type
        self.data_for_analize = data
        self.start()

    def __call__(self, *args, **kwargs):
        print(self.RESULT)

    def normalize(self, values, param_name):  # нормализация значений
        # Запись данных параметра в бд
        connector = connectToDB()
        updateParamToDB(connector, {"data": {
            "param": param_name,
            "min": min(values),
            "max": max(values)
        }})
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

    def normalizeInput(self, value, param_name):  # нормализация значений
        # Запись данных параметра в бд
        connector = connectToDB()
        values = getParamInputValueForNormalize(connector, param_name)
        x = float(value)
        x_min = values['min']
        x_max = values['max']
        b = 1
        a = 0
        y = ((x - x_min) / (x_max - x_min)) * (b - a) + a
        print('x-min is %s, x-max is %s, x is %s, y is %s, param_name is %s' % (x_min, x_max, x, y, param_name))
        return y

    def denormalize(self, x, param):  # денормализация значений
        connector = connectToDB()
        values = getParamValuesFromDB(connector, param)
        x_min = values['min']
        x_max = values['max']
        y = (x_max - x_min) * x + x_min
        print("y = ", y)
        return y

    def save_data(self, net, sample_name):  # сохранение сети в файл
        file_name = '{0}'.format(str(sample_name))
        NetworkWriter.writeToFile(net, file_name + '.xml')
        print("File saved with name '{0}'.".format(str(file_name)))

    def load_data(self, sample_name):  # загрузка сети из файла
        net = NetworkReader.readFrom(sample_name + '.xml')
        return net

    def get_data(self, filename):  # получение первичных данных из csv файла
        connector = connectToDB()
        formatted_data = get_all_users(connector)
        data_set = SupervisedDataSet(10, 1)

        for row in formatted_data:
            self.sexes.append(row[1])
            self.ages.append(row[2])
            self.heights.append(row[3])
            self.body_mass.append(row[4])
            self.chests.append(row[5])
            self.body_index_masses.append(row[6])
            self.shoulders.append(row[7])
            self.forearms.append(row[8])
            self.shins.append(row[9])
            self.leans.append(row[10])
            self.mep.append(row[11])
            self.mip.append(row[12])
            self.snip.append(row[13])

        self.normalized_sexes = self.normalize(self.sexes, "sex")
        self.normalized_ages = self.normalize(self.ages, "age")
        self.normalized_heights = self.normalize(self.heights, "height")
        self.normalized_body_mass = self.normalize(self.body_mass, "bm")
        self.normalized_chests = self.normalize(self.chests, "chest")
        self.normalized_body_index_masses = self.normalize(self.body_index_masses, "bim")
        self.normalized_shoulders = self.normalize(self.shoulders, "shoulder")
        self.normalized_forearms = self.normalize(self.forearms, "forearm")
        self.normalized_shins = self.normalize(self.shins, "shin")
        self.normalized_leans = self.normalize(self.leans, "lean")
        self.normalized_mep = self.normalize(self.mep, "mep")
        self.normalized_mip = self.normalize(self.mip, "mip")
        self.normalized_snip = self.normalize(self.snip, "snip")
        print("self.normalized_sexes", self.normalized_sexes)
        print("self.normalized_ages", self.normalized_ages)
        print("self.normalized_shoulders", self.normalized_shoulders)
        print("self.normalized_heights", self.normalized_heights)
        print("self.normalized_chests", self.normalized_chests)
        print("self.normalized_body_index_masses", self.normalized_body_index_masses)
        print("self.normalized_body_mass", self.normalized_body_mass)
        print("self.normalized_leans", self.normalized_leans)
        print("self.normalized_forearms", self.normalized_forearms)
        print("self.normalized_shins", self.normalized_shins)
        print("self.normalized_mep", self.normalized_mep)
        print("self.normalized_mip", self.normalized_mip)
        print("self.normalized_snip", self.normalized_snip)

        # Название типа анализа
        sample = self.sample_name
        current_type_array = []
        if sample is 'mep': current_type_array = self.normalized_mep['out']['data']
        elif sample is 'mip': current_type_array = self.normalized_mip['out']['data']
        elif sample is 'snip': current_type_array = self.normalized_snip['out']['data']

        for sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean, output in zip(
                self.normalized_sexes['out']['data'],
                self.normalized_ages['out']['data'],
                self.normalized_heights['out']['data'],
                self.normalized_body_mass['out']['data'],
                self.normalized_chests['out']['data'],
                self.normalized_body_index_masses['out']['data'],
                self.normalized_shoulders['out']['data'],
                self.normalized_forearms['out']['data'],
                self.normalized_shins['out']['data'],
                self.normalized_leans['out']['data'],
                current_type_array):
            sample = (sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean), output
            print("sample", sample)
            data_set.addSample((sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean), output)

        return data_set

    def get_test_learned_data(self, file_type):  # получение первичных данных из csv файла
        if file_type is 1:
            filename = "data.csv"
        elif file_type is 0:
            filename = "data_test.csv"
        else:
            return None
        doc = open(filename, 'rb')
        reader = csv.reader(doc)
        formatted_data = []
        for row in reader:
            for items in row:
                splitted_items = items.split(';')
                floated_items = [(float(elem)) for elem in splitted_items]
                formatted_data.append(floated_items)
        for row in formatted_data:
            self.sexes_test.append(row[0])
            self.ages_test.append(row[1])
            self.heights_test.append(row[2])
            self.body_mass_test.append(row[3])
            self.chests_test.append(row[4])
            self.body_index_masses_test.append(row[5])
            self.shoulders_test.append(row[6])
            self.forearms_test.append(row[7])
            self.shins_test.append(row[8])
            self.leans_test.append(row[9])
            self.mep_test.append(row[10])
            self.mip_test.append(row[11])
            self.snip_test.append(row[12])

        samples = {
            "name": '',
            "data": []
        }

        # Название типа анализа
        sample = str(self.sample_name)
        print('sample is %s and type is %s' % (sample, type(sample)))
        current_type_array = []
        if sample == 'mep':
            current_type_array = self.mep_test
            print('sample is %s' % sample)
        elif sample == 'mip':
            current_type_array = self.mip_test
            print('sample is %s' % sample)
        elif sample == 'snip':
            current_type_array = self.snip_test
            print('sample is %s' % sample)

        print('current_type_array test', current_type_array)

        for sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean, output in zip(
                self.sexes_test,
                self.ages_test,
                self.heights_test,
                self.body_mass_test,
                self.chests_test,
                self.body_index_masses_test,
                self.shoulders_test,
                self.forearms_test,
                self.shins_test,
                self.leans_test,
                current_type_array):
            sample = (sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean), output
            print('get_test_learned_data', sample)
            samples['data'].append(sample)
            print("%s test" % filename, sample)
        samples['name'] = filename
        return samples

    def create_neural_net(self):
        rand_value = 26015 # TODO: посмотреть еще, зачем оно надо...
        # Создание сети
        net = FeedForwardNetwork()
        # Параметры сети
        inp = LinearLayer(10)
        out = LinearLayer(1)
        hidden1 = SigmoidLayer(13)
        hidden2 = TanhLayer(8)
        hidden3 = TanhLayer(6)
        hidden4 = TanhLayer(6)
        # Модули сети
        net.addOutputModule(out)
        net.addInputModule(inp)
        net.addModule(hidden1)
        net.addModule(hidden2)
        net.addModule(hidden3)
        net.addModule(hidden4)
        # Создание связей
        net.addConnection(FullConnection(inp, hidden1))
        net.addConnection(FullConnection(hidden1, hidden2))
        net.addConnection(FullConnection(hidden2, hidden3))
        net.addConnection(FullConnection(hidden3, hidden4))
        net.addConnection(FullConnection(hidden4, out))
        # Подготовка - сортировка модулей
        net.sortModules()
        return net

    def train_net(self, data_set, epochs): # тренировка данных
        net = self.create_neural_net()
        trainer = BackpropTrainer(net, data_set)
        print('BackpropTrainer DONE')
        for i in range(0, epochs):
            if i % 10 is 0 and i is not 0:
                print('Тренировка преодолела рубеж -> %s шагов' % i)
            trainer.train()
        print('TrainUntilConvergence DONE')
        return net

    def get_result(self, net):
        data = []
        sex = self.normalizeInput(self.data_for_analize[0], "sex")
        age = self.normalizeInput(self.data_for_analize[1], "age")
        height = self.normalizeInput(self.data_for_analize[2], "height")
        bm = self.normalizeInput(self.data_for_analize[3], "bm")
        chest = self.normalizeInput(self.data_for_analize[4], "chest")
        bim = self.normalizeInput(self.data_for_analize[5], "bim")
        shoulder = self.normalizeInput(self.data_for_analize[6], "shoulder")
        forearm = self.normalizeInput(self.data_for_analize[7], "forearm")
        shin = self.normalizeInput(self.data_for_analize[8], "shin")
        lean = self.normalizeInput(self.data_for_analize[9], "lean")
        data.append(sex)
        data.append(age)
        data.append(height)
        data.append(bm)
        data.append(chest)
        data.append(bim)
        data.append(shoulder)
        data.append(forearm)
        data.append(shin)
        data.append(lean)
        print('data', data)
        answer = net.activate(data)
        print("answer", answer)
        return self.denormalize(answer, self.sample_name)

    def get_result_test(self, net):
        learned_report = []
        tested_report = []
        learned_errors = []
        tested_errors = []
        # Полученные данные обучающая и тестовая выборки
        learned_data = self.get_test_learned_data(1)
        test_data = self.get_test_learned_data(0)
        print('get_result_test', learned_data, test_data)
        # Выбираем входные данные
        learned_data_inputs = [list(item[0][0:11]) for item in learned_data['data']]
        test_data_inputs = [list(item[0][0:11]) for item in test_data['data']]
        # Выбираем ответы
        learned_data_answers = [item[1] for item in learned_data['data']]
        test_data_answers = [item[1] for item in test_data['data']]
        # Разбор данных обучающая выборка
        for i in range(len(learned_data_inputs)):
            results = {
                'type': 'learned',
                'input': [],
                'expected': 0,
                'real': 0,
                'error': 0
            }
            data = []
            input_data = learned_data_inputs[i]
            print('learned_data_inputs[i]', learned_data_inputs[i])
            answer_data = learned_data_answers[i]
            sex = self.normalizeInput(input_data[0], "sex")
            age = self.normalizeInput(input_data[1], "age")
            height = self.normalizeInput(input_data[2], "height")
            bm = self.normalizeInput(input_data[3], "bm")
            chest = self.normalizeInput(input_data[4], "chest")
            bim = self.normalizeInput(input_data[5], "bim")
            shoulder = self.normalizeInput(input_data[6], "shoulder")
            forearm = self.normalizeInput(input_data[7], "forearm")
            shin = self.normalizeInput(input_data[8], "shin")
            lean = self.normalizeInput(input_data[9], "lean")
            data.append(sex)
            data.append(age)
            data.append(height)
            data.append(bm)
            data.append(chest)
            data.append(bim)
            data.append(shoulder)
            data.append(forearm)
            data.append(shin)
            data.append(lean)
            print('data get_result_test', data)
            norm_answer = net.activate(data)
            answer = self.denormalize(norm_answer, self.sample_name)
            results['input'] = input_data
            results['expected'] = answer_data
            results['real'] = answer
            results['error'] = calculate_error(answer_data, answer)
            learned_errors.append(results['error'])
            learned_report.append(results)
        for row in learned_report:
            print(row)
        print('learned mean = ', mean(learned_errors))
        # Разбор данных тестовая выборка
        for i in range(len(test_data_inputs)):
            results = {
                'type': 'tested',
                'input': [],
                'expected': 0,
                'real': 0,
                'error': 0
            }
            data = []
            input_data = test_data_inputs[i]
            answer_data = test_data_answers[i]
            sex = self.normalizeInput(input_data[0], "sex")
            age = self.normalizeInput(input_data[1], "age")
            height = self.normalizeInput(input_data[2], "height")
            bm = self.normalizeInput(input_data[3], "bm")
            chest = self.normalizeInput(input_data[4], "chest")
            bim = self.normalizeInput(input_data[5], "bim")
            shoulder = self.normalizeInput(input_data[6], "shoulder")
            forearm = self.normalizeInput(input_data[7], "forearm")
            shin = self.normalizeInput(input_data[8], "shin")
            lean = self.normalizeInput(input_data[9], "lean")
            data.append(sex)
            data.append(age)
            data.append(height)
            data.append(bm)
            data.append(chest)
            data.append(bim)
            data.append(shoulder)
            data.append(forearm)
            data.append(shin)
            data.append(lean)
            norm_answer = net.activate(data)
            answer = self.denormalize(norm_answer, str(self.sample_name))
            print('answer net', answer)
            results['input'] = input_data
            results['expected'] = answer_data
            results['real'] = answer
            results['error'] = calculate_error(answer_data, answer)
            tested_errors.append(results['error'])
            tested_report.append(results)
        for row in tested_report:
            print(row)
        print('tested mean = ', mean(tested_errors))
        # return learned_data_answers

    def start(self):  # запуск приложения
        if self.query_type == 'train':
            # Получение данных из csv файла
            data_set = self.get_data(self.file_name)
            # Тренировка на данных и получение тренированной сети
            net = self.train_net(data_set, self.EPOCHS)
            # Сохранение нейросети в файл
            self.save_data(net, self.sample_name)
        elif self.query_type == 'get_answer':
            # Загрузка нейросети из файла
            net = self.load_data(self.sample_name)
            # Активация и получение результата
            self.RESULT = self.get_result(net)
            self.get_result_test(net)
            # print('Returned report ', report)
