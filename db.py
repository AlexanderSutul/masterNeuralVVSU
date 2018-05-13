# coding=utf-8
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
