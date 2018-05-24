# coding=utf-8
from flask import Flask, request
from flask_cors import CORS
import json

from classes import NeuralNetMaster
from db import *

app = Flask(__name__)
CORS(app)

# Добавление всех пациентов в БД
# @app.route("/add_patients", methods=["GET"])
# def add_patients():
#     connector = connectToDB()
#     result = add_pattients_to_db_from_csv(connector, 'data.csv')
#     return result

# Добавление здорового пациента в БД Тест
@app.route("/add_patient_test", methods=["GET"])
def add_patient_test():
    data = ['null', 0, 18, 1.64, 60, 78, 22.30814991, 35, 27,	37,	35696.20969, 69, 55, 55]
    connector = connectToDB()
    result = add_new_patient(connector, data)
    return result

# Получить всех пользователей
@app.route("/getPatients", methods=["GET"])
def get_patients():
    return json.dumps(get_all_users(connectToDB()))

# Удалить пользователя
@app.route("/deletePatient", methods=["POST"])
def delete_patients():
    post = request.get_json(silent=True)
    id = post['deletedPatient']
    delete_patient(connectToDB(), id)
    return json.dumps(get_all_users(connectToDB()))

# Добавление здорового пациента в БД
@app.route("/addPatient", methods=["POST"])
def add_patient():
    post = request.get_json(silent=True)
    print('post', post)
    data = ['null', post['sex'], post['age'], post['height'], post['bodyMass'],
            post['chest'], post['bodyMassIndex'], post['shoulder'],
            post['forearm'], post['shin'], post['lean'],
            post['mep'], post['mip'], post['snip']]
    connector = connectToDB()
    result = add_new_patient(connector, data)
    return json.dumps(get_all_users(connectToDB()))

# Получение ответа
@app.route("/answer", methods=["POST"])
def get_answer():
    post = request.get_json(silent=True)
    data = [post['sex'], post['age'], post['height'], post['bodyMass'],
            post['chest'], post['bodyMassIndex'], post['shoulder'],
            post['forearm'], post['shin'], post['lean']]
    query_type = 'get_answer'
    type_for_analize = post['type']
    answer = NeuralNetMaster("data.csv", type_for_analize, query_type, data)
    return str(answer.RESULT)

# Обучение нейронной сети
@app.route("/train_mep", methods=["GET"])
def send_for_train_mep():
    NeuralNetMaster("data.csv", 'mep', 'train', None)
    return "Обучение 'mep' прошло успешно!\n"

@app.route("/train_mip", methods=["GET"])
def send_for_train_mip():
    NeuralNetMaster("data.csv", 'mip', 'train', None)
    return "Обучение 'mip' прошло успешно!\n"

@app.route("/train_snip", methods=["GET"])
def send_for_train_snip():
    NeuralNetMaster("data.csv", 'snip', 'train', None)
    return "Обучение 'snip' прошло успешно!\n"

# Тестирование, что все работает
@app.route("/test", methods=["GET"])
def test():
    return "Flask is working.\n"

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
