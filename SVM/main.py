import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import random

data = [[i for i in i.strip().split(",")] for i in open("./abalone.data").readlines()]

# Format strings to floats/ints, sort into correct list
m_abalones, f_abalones, i_abalones = [], [], []
for abalone in data:
    abl = []
    for i in range(1, len(abalone)):
        abl.append(float(abalone[i]))
        if i == 8:
            abl[i - 1] = int(abl[i - 1])
    if abalone[0] == 'M':
        m_abalones.append(abl)
    elif abalone[0] == 'F':
        f_abalones.append(abl)
    else:
        i_abalones.append(abl)


# Shuffle the lists for anonymity
random.shuffle(m_abalones)
random.shuffle(f_abalones)
random.shuffle(i_abalones)

# Split data into train and test lists
training_m = [i for i in m_abalones[:int(len(m_abalones)/2)]]
training_f = [i for i in f_abalones[:int(len(f_abalones)/2)]]
training_i = [i for i in i_abalones[:int(len(i_abalones)/2)]]
testing_m = [i for i in m_abalones[int(len(m_abalones)/2):]]
testing_f = [i for i in f_abalones[int(len(f_abalones)/2):]]
testing_i = [i for i in i_abalones[int(len(i_abalones)/2):]]


def mapTo2D(data):
    retval = []
    for i in range(0, len(data), 2):
        x = data[i]
        y = data[i + 1]
        retval.append((x, y))
    return retval


# Sort data into 2D lists
training_2d_m, training_2d_f, training_2d_i = [], [], []
for m, f, i in zip(training_m, training_f, training_i):
    training_2d_m += mapTo2D(m)
    training_2d_f += mapTo2D(f)
    training_2d_i += mapTo2D(i)


# PLOTTING 2D
training_2d_sets = [training_2d_m, training_2d_f, training_2d_i]
for index, training in enumerate(training_2d_sets, 1):
    plt.subplot(2, 3, index)
    plt.plot([i[0] for i in training], [i[1] for i in training], "-o", color="green")

for index, training in enumerate(training_2d_sets, 4):
    plt.subplot(2, 3, index)
    plt.plot([i[0] for i in training][:4], [i[1] for i in training][:4], "-o", color="green")

plt.show()


for index, training in enumerate(training_2d_sets):
    title = "Male" if index == 0 else ("Female" if index == 1 else "Infant")
    color = "green" if index == 0 else ("orange" if index == 1 else "blue")
    start = 0
    end = 4
    for k in range(1, 4):
        plt.subplot(3, 3, (3 * index) + k, title=f"{title} {k}")
        plt.plot([i[0] for i in training][start:end], [i[1] for i in training][start:end], "-o", color=color)
        start += 4
        end += 4

plt.show()

# VARIABLES
C = 1.0
gamma = 0.5
h = 0.2

names = ["Linear",
         "Gaussian",
         "Sigmoid",
         # "Poly",
         ]

classifiers = [svm.SVC(kernel="linear", C=C, gamma=gamma),
               svm.SVC(kernel="rbf", C=C, gamma=gamma),
               svm.SVC(kernel="sigmoid", C=C, gamma=gamma),
               # svm.SVC(kernel="poly", C=C, gamma=gamma),
               ]


# 2D classification
X = np.array(training_2d_m + training_2d_f + training_2d_i)
Y = np.array([0 for i in training_2d_m] + [1 for i in training_2d_f] + [2 for i in training_2d_i])

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


def plotSVM(svm, n, title):
    plt.subplot(2, 3, n)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.title(title)


def testSVM(svm, male, female, infant):
    num_correct = 0
    num_wrong = 0
    for correct, testing in ((0, male), (1, female), (2, infant)):
        for d in testing:
            npd = np.array(d).reshape(1, -1)
            r = svm.predict(npd)[0]
            if r == correct:
                num_correct += 1
            else:
                num_wrong += 1
    print("\t=> Correct:", num_correct)
    print("\t=> Wrong:", num_wrong)
    print("\t=> Accuracy:", float(num_correct) / (num_correct + num_wrong))


testing_2d_m, testing_2d_f, testing_2d_i = [], [], []
for m, f, i in zip(testing_m, testing_f, testing_i):
    testing_2d_m += mapTo2D(m)
    testing_2d_f += mapTo2D(f)
    testing_2d_i += mapTo2D(i)


print("Testing 2D\n**********")
for i, classifier in enumerate(classifiers):
    classifier.fit(X, Y)
    print("\n" + names[i])
    testSVM(classifier, testing_2d_m, testing_2d_f, testing_2d_i)
    # plotSVM(classifier, i + 1, names[i])
# plt.show()


# 16D CLASSIFICATION
X = np.array(training_m + training_f + training_i)
Y = np.array([0 for i in training_m] + [1 for i in training_f] + [2 for i in training_i])

print("\n\nTesting 16D\n***********")
for i, classifier in enumerate(classifiers):
    classifier.fit(X, Y)
    print("\n" + names[i])
    testSVM(classifier, testing_m, testing_f, testing_i)

# Plot 16D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

