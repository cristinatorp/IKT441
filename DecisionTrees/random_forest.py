import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x = []
y = []
patients = csv.reader(open("./breast-cancer.data", "r"))
all_patients = []


def get_recurrence(p_class):
    try:
        if p_class == "no-recurrence-events":
            return 1
    except ValueError:
        pass
    return 0


# Assumption: 10-49 = young (30), else old (70)
def get_old_young(age):
    try:
        if int(age[0]) <= 4:
            return 30
    except ValueError:
        pass
    return 70


# Assumption: 0-14 = small (10), 15-39 = medium (25), 40-59 = large (50)
def get_tumor_size(size):
    try:
        if int(size[0]) < 2:
            return 10
        elif int(size[0]) < 4:
            return 25
    except ValueError:
        pass
    return 50


# Recurrence, age, tumor-size, deg-malig
for patient in patients:
    x.append([get_old_young(patient[1]), get_tumor_size(patient[3]), patient[6]])
    y.append(get_recurrence(patient[0]))


x_tmp = np.array(x)
y_tmp = np.array(y)
x = x_tmp.reshape(x_tmp.shape[0], -1)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)

rf = RandomForestClassifier()

rf.fit(x_train, y_train)
y_predict = rf.predict(x_test)
total = x_test.shape[0]
accuracy = (100 * ((y_test == y_predict).sum() / total))
print("Accuracy =", round(accuracy, 2), "%")

