#!/usr/bin/python
# coding=utf-8
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

import csv

# БД
import _sqlite3 as sqlite


def connectToDB():
    connector = sqlite.connect('neural_db.db')
    return connector


def updateParamToDB(connector, data):
    cursor = connector.cursor()
    param_name = data['data']['param']
    minimal = data['data']['min']
    maximum = data["data"]["max"]
    sql = """
            UPDATE params 
            SET min = %s, max = %s 
            WHERE param = '%s'
            """ % (minimal, maximum, param_name)
    print(sql)
    cursor.execute(sql)
    connector.commit()
    connector.close()


def getParamValuesFromDB(connector, param):
    result_object = {
        "min": 0,
        "max": 0
    }
    sql = "SELECT * FROM params WHERE param = '%s'" % param
    cursor = connector.cursor()
    cursor.execute(sql)
    record = cursor.fetchone()
    result_object['min'] = record[2]
    result_object['max'] = record[3]
    print("data getParamValuesFromDB", result_object)
    return result_object


def getParamInputValueForNormalize(connector, param):
    result_object = {
        "min": 0,
        "max": 0
    }
    sql = "SELECT * FROM params WHERE param = '%s'" % param
    cursor = connector.cursor()
    cursor.execute(sql)
    record = cursor.fetchone()
    result_object['min'] = record[2]
    result_object['max'] = record[3]
    print("getParamInputValueForNormalize", result_object)
    return result_object


class NeuralNetMaster:
    EPOCHS = 10000
    VALID_PROP = 0.99
    VERBOSE = True
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
    out = []

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
    out_test = []

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

    normalized_sexes_test = []
    normalized_ages_test = []
    normalized_shoulders_test = []
    normalized_heights_test = []
    normalized_chests_test = []
    normalized_body_index_masses_test = []
    normalized_body_mass_test = []
    normalized_leans_test = []
    normalized_forearms_test = []
    normalized_shins_test = []
    normalized_out_test = []

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
        print(x_min, x_max, x, y, param_name)
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
                floated_items = [(float(elem)) for elem in splitted_items]
                formatted_data.append(floated_items)
        data_set = SupervisedDataSet(10, 1)

        for row in formatted_data:
            self.sexes.append(row[0])
            self.ages.append(row[1])
            self.heights.append(row[2])
            self.body_mass.append(row[3])
            self.chests.append(row[4])
            self.body_index_masses.append(row[5])
            self.shoulders.append(row[6])
            self.forearms.append(row[7])
            self.shins.append(row[8])
            self.leans.append(row[9])
            self.out.append(row[10])

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
        self.normalized_out = self.normalize(self.out, "out")
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
        print("self.normalized_out", self.normalized_out)

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
                self.normalized_out['out']['data']):
            sample = (sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean), output
            print("sample", sample)
            data_set.addSample((sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean), output)

        return data_set

    def get_test_data(self, filename):  # получение первичных данных из csv файла
        doc = open(filename, 'rb')
        reader = csv.reader(doc)
        formatted_data = []
        for row in reader:
            for items in row:
                splitted_items = items.split(';')
                floated_items = [(float(elem)) for elem in splitted_items]
                formatted_data.append(floated_items)
        data_set = SupervisedDataSet(10, 1)

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
            self.out_test.append(row[10])

        self.normalized_sexes_test = self.normalize(self.sexes_test, "sex")
        self.normalized_ages_test = self.normalize(self.ages_test, "age")
        self.normalized_heights_test = self.normalize(self.heights_test, "height")
        self.normalized_body_mass_test = self.normalize(self.body_mass_test, "bm")
        self.normalized_chests_test = self.normalize(self.chests_test, "chest")
        self.normalized_body_index_masses_test = self.normalize(self.body_index_masses_test, "bim")
        self.normalized_shoulders_test = self.normalize(self.shoulders_test, "shoulder")
        self.normalized_forearms_test = self.normalize(self.forearms_test, "forearm")
        self.normalized_shins_test = self.normalize(self.shins_test, "shin")
        self.normalized_leans_test = self.normalize(self.leans_test, "lean")
        self.normalized_out_test = self.normalize(self.out_test, "out")
        print("self.normalized_sexes", self.normalized_sexes_test)
        print("self.normalized_ages", self.normalized_ages_test)
        print("self.normalized_shoulders", self.normalized_shoulders_test)
        print("self.normalized_heights", self.normalized_heights_test)
        print("self.normalized_chests", self.normalized_chests_test)
        print("self.normalized_body_index_masses", self.normalized_body_index_masses_test)
        print("self.normalized_body_mass", self.normalized_body_mass_test)
        print("self.normalized_leans", self.normalized_leans_test)
        print("self.normalized_forearms", self.normalized_forearms_test)
        print("self.normalized_shins", self.normalized_shins_test)
        print("self.normalized_out", self.normalized_out_test)

        for sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean, output in zip(
                self.normalized_sexes_test['out']['data'],
                self.normalized_ages_test['out']['data'],
                self.normalized_heights_test['out']['data'],
                self.normalized_body_mass_test['out']['data'],
                self.normalized_chests_test['out']['data'],
                self.normalized_body_index_masses_test['out']['data'],
                self.normalized_shoulders_test['out']['data'],
                self.normalized_forearms_test['out']['data'],
                self.normalized_shins_test['out']['data'],
                self.normalized_leans_test['out']['data'],
                self.normalized_out_test['out']['data']):
            sample = (sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean), output
            print("sample test", sample)
            data_set.addSample((sex, age, height, bm, chest, bim, shoulder, forearm, shin, lean), output)

        return data_set

    def test_answer(self):
        
        pass

    def train_net(self, data_set, epochs, validProp, verbose):  # тренировка данных
        net = buildNetwork(10, 3, 1)
        trainer = BackpropTrainer(net, data_set)
        trainer.trainUntilConvergence(maxEpochs=epochs,
                                      verbose=verbose)
        self.get_test_data('mep_test_data_last.csv')
        return net

    def get_result(self, net):
        # Примеры для проверки
        # input_data1 = [0.0, 0.3287671232876712, 0.25789786664684455, 0.47761194029850745, 0.5882352941176471,
        #                0.5318417604869441, 0.3, 0.4117647058823529, 0.45, 0.3453618706781417]  # 1.41939898
        # input_data2 = [1.0, 0.1780821917808219, 1.0, 0.5223880597014925, 0.29411764705882354, 0.2248854027747037, 0.35,
        #                0.35294117647058826, 0.4, 0.40165179717288246]  # 0.18846153846153846
        print("self.data_for_analize", self.data_for_analize)
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
        return self.denormalize(answer, "out")

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
            self.RESULT = self.get_result(net)

# TODO Сделать проверку на ошибку, достаточно иметь 20%, дальше не лезть!
