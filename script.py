# coding=utf-8
from flask import Flask, request
from flask_cors import CORS

from neuralClass import NeuralNetMaster

app = Flask(__name__)
CORS(app)


@app.route("/answer", methods=["GET", "POST"])
def get_answer():
    age = request.args.get("age")

    print(request.args)

    # Начало работы программы
    # input_data = [1, 19, 31048, 70, 85, 20.45288532, 37, 26, 34, 39714.94834]  # 132.87976709
    input_data = [0, 48, 23377, 80, 90, 29.74419988, 35, 29, 33, 36685.23509]  # 87.30399729
    # Запросы:
    # get_data_from_csv_file_and_train => Получение данных из файла CSV и тренировка данных и Сохранение
    # тренированной сети в файле
    # get_answer => Загрузка тренированной сети из файла и получение ответа
    query_type = 'get_answer'

    answer = NeuralNetMaster("mep_test_data_last.csv", 'mep', query_type)
    print(answer.RESULT)
    try:
        return str(answer.RESULT[0])
    except ValueError:
        return "some error"


@app.route("/train", methods=["GET", "POST"])
def send_for_train():
    age = request.args.get("age")

    print(request.args)

    # Начало работы программы
    # input_data = [1, 19, 31048, 70, 85, 20.45288532, 37, 26, 34, 39714.94834]  # 1.41939898
    # input_data = [0, 48, 23377, 80, 90, 29.74419988, 35, 29, 33, 36685.23509]  # 0.79700267
    # Запросы:
    # get_data_from_csv_file_and_train => Получение данных из файла CSV и тренировка данных и Сохранение
    # тренированной сети в файле
    # get_answer => Загрузка тренированной сети из файла и получение ответа
    query_type = 'get_data_from_csv_file_and_train'

    answer = NeuralNetMaster("mep_test_data_last.csv", 'mep', query_type)
    print(answer.RESULT)
    return "Вроде, работает"


if __name__ == '__main__':
    app.run(debug=False)
