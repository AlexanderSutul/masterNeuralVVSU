# coding=utf-8
import _sqlite3 as sqlite
import csv

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
    if param_name is 'snip':
        print('<------ALARM is %s->>>>>>>>>>' % param_name)
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

def add_new_patient(connector, input_data):
    new_patient_params = ", ".join(map(str, input_data))
    sql = "INSERT INTO patients VALUES (%s)" % new_patient_params
    cursor = connector.cursor()
    result = cursor.execute(sql)
    connector.commit()
    connector.close()
    return 'Patient added.'

def delete_patient(id):
    return 'Patient with id -> %s has been deleted.\n' % id

def update_patient(id, data):
    return 'Patient with id -> %s has been edited. \n' % (id, data)

def add_pattients_to_db_from_csv(connector, filename):
    doc = open(filename, 'rb')
    reader = csv.reader(doc)
    formatted_data = []
    cursor = connector.cursor()
    for row in reader:
        for items in row:
            splitted_items = items.split(';')
            floated_items = [float(elem) for elem in splitted_items]
            formatted_data.append(floated_items)
    for patient in formatted_data:
        patient_params = ", ".join(map(str, patient))
        sql_query = "INSERT INTO patients VALUES (null, %s)" % patient_params
        print(sql_query)
        cursor.execute(sql_query)
    connector.commit()
    connector.close()
    return 'All data added to DB'

def get_all_users(connector):
    users = []
    cursor = connector.cursor()
    sql_query = "SELECT * FROM patients ORDER BY id DESC"
    cursor.execute(sql_query)
    users = cursor.fetchall()
    connector.close()
    return users

def delete_patient(connector, id):
    cursor = connector.cursor()
    sql_query = "DELETE FROM patients WHERE id=%s" % id
    cursor.execute(sql_query)
    connector.commit()
    connector.close()
    return 'Patient with id %s was deleted' % id
