import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, accuracy_score, f1_score

classifier = DecisionTreeClassifier()
training_file_name = 'data/old/train_normalized.csv'
testing_file_name = 'data/old/test_normalized.csv'

R = 0.09
np = 64
number_of_iterations = 200
U = 1
L = 0



def get_number_of_inputs():
    global number_of_inputs
    return number_of_inputs


def init():
    global data_train, data_test, target_train, target_test, number_of_inputs

    

    train_set = pd.read_csv(training_file_name)
    test_set = pd.read_csv(testing_file_name)


    number_of_inputs = len(train_set.columns) - 1

    data_train = train_set.iloc[:, :number_of_inputs]
    target_train = train_set.iloc[:, number_of_inputs]

    data_test = test_set.iloc[:, :number_of_inputs]
    target_test = test_set.iloc[:, number_of_inputs]


def calc_fitness(px):
    select = []
    for i, xi in enumerate(px):
        if xi >= .5:
            select.append(i)
    if len(select) == 0:
        return 0, 1, 0

    td = data_train.iloc[:, select]
    dt = data_test.iloc[:, select]


    pred = classifier.fit(td, target_train).predict(dt)
    r = recall_score(target_test, pred, average=None)

    return r[0], (1 - r[1]), len(select)


def acc__f_score(px):
    select = []
    for i, xi in enumerate(px):
        if xi >= .5:
            select.append(i)
    if len(select) == 0:
        return 0, 1, 0

    td = data_train.iloc[:, select]
    dt = data_test.iloc[:, select]

    pred = classifier.fit(td, target_train).predict(dt)
    acc = accuracy_score(target_test, pred)
    f_score = f1_score(target_test, pred, average='macro')
    return acc, f_score


def get_attr(px):
    select = []
    for i, xi in enumerate(px):
        if xi >= .5:
            select.append(i)
    return select
