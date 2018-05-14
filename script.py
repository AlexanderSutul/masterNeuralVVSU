# coding=utf-8
from flask import Flask, request
from flask_cors import CORS

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

# # Добавление здорового пациента в БД
# @app.route("/add_patient", methods=["GET"])
# def add_patient():
#     post = request.get_json(silent=True)
#     # data = [post['sex'], post['age'], post['height'], post['bodyMass'],
#     #         post['chest'], post['bodyMassIndex'], post['shoulder'],
#     #         post['forearm'], post['shin'], post['lean'], post['out']]
#     data = ['null', 0, 19, 1.62, 60, 78, 22.86236854, 36, 27, 36, 33820.475, 70, 70, 70]
#     connector = connectToDB()
#     result = add_new_patient(connector, data)
#     return result

# Получение ответа
@app.route("/answer", methods=["POST"])
def get_answer():
    post = request.get_json(silent=True)
    data = [post['sex'], post['age'], post['height'], post['bodyMass'],
            post['chest'], post['bodyMassIndex'], post['shoulder'],
            post['forearm'], post['shin'], post['lean']]
    query_type = 'get_answer'
    type_for_analize = post['type']
    answer = NeuralNetMaster("mep_data_last.csv", type_for_analize, query_type, data)
    return str(answer.RESULT)

# Начало работы программы
@app.route("/train", methods=["GET"])
def send_for_train():
    query_type = 'get_data_from_csv_file_and_train'
    train_type = 'snip'
    NeuralNetMaster("data.csv", train_type, query_type, None)
    return "Обучение прошло успешно!\n"

# Проверка, что все работает
@app.route("/test", methods=["GET"])
def test():
    return "Flask is working.\n"

if __name__ == '__main__':
    app.run(debug=False)
