import csv, math, random

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   1. Predict the two classes                              #
#   2. With only Python, and with sklearn (random forest)   #
#   3. Train and validate all algorithms                    #
#   4. Make the necessary assumptions                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

""" 
ATTRIBUTES OVERVIEW (* = Missing values)

1. Class (non-recurrent, recurrent)
2. Age (10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99)
3. Menopause (lt40, ge40, premeno)
4. Tumor-size (0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59)
5. Inv-nodes (0-2, 3-5, 6-8, 9-11, 12-14, 15-17, 18-20, 21-23, 24-26, 27-29, 30-32, 33-35, 36-39)
6. * Node-caps (yes, no)
7. Deg-malig (1, 2, 3)
8. Breast (left, right)
9. Breast-quad (left-up, left-low, right-up, right-low, central)
10. * Irradiat (yes, no)
"""
patients = csv.reader(open("./breast-cancer.data", "r"))
no_recurrence_events = []
recurrence_events = []


def get_recurrence(p_class):
    try:
        if p_class == "no-recurrence-events":
            return "not recurring"
    except ValueError:
        pass
    return "recurring"


# Assumption: 10-49 = young, else old
def get_old_young(age):
    try:
        if int(age[0]) <= 4:
            return "young"
    except ValueError:
        pass
    return "old"


# Assumption: 0-14 = small, 15-39 = medium, 40-59 = large
def get_tumor_size(size):
    try:
        if int(size[0]) < 2:
            return "small"
        elif int(size[0]) < 4:
            return "medium"
    except ValueError:
        pass
    return "large"


def get_degree(deg_malig):
    try:
        if int(deg_malig) == 1:
            return "degree 1"
        elif int(deg_malig) == 2:
            return "degree 2"
    except ValueError:
        pass
    return "degree 3"


# Recurrence, age, tumor-size, deg-malig
for patient in patients:
    if patient[0] == "no-recurrence-events":
        no_recurrence_events.append([get_recurrence(patient[0]),
                                     get_old_young(patient[1]),
                                     get_tumor_size(patient[3]),
                                     get_degree(patient[6])])
    else:
        recurrence_events.append([get_recurrence(patient[0]),
                                  get_old_young(patient[1]),
                                  get_tumor_size(patient[3]),
                                  get_degree(patient[6])])

training_data = no_recurrence_events[:int(len(no_recurrence_events)/2)] + recurrence_events[:int(len(recurrence_events)/2)]
verification_data = no_recurrence_events[int(len(no_recurrence_events)/2):] + recurrence_events[int(len(recurrence_events)/2):]
random.shuffle(training_data)
random.shuffle(verification_data)


def entropy(one_class):
    pos = len([i for i in one_class if i[0] == "recurring"])
    neg = len([i for i in one_class if i[0] == "not recurring"])
    total = pos + neg

    if min((pos, neg)) == 0:
        return 0

    entropy = - (pos / total) * math.log(pos / total, 2) - (neg / total) * math.log(neg / total, 2)
    return entropy


def split(data, attribute, remove=False):
    retvals = {}

    for d in data:
        c = d[attribute]
        a_list = retvals.get(c, [])
        if remove:
            d.pop(attribute)
        a_list.append(d)
        retvals[c] = a_list

    return retvals


def get_highest_gain(one_class):
    classes = [i for i in range(1, len(one_class[0]))]
    entropies = [gain(one_class, c) for c in classes]
    return entropies.index(min(entropies)) + 1


def is_pure(one_class):
    classes = [i for i in range(1, len(one_class[0]))]
    for c in classes:
        if len(set([i[c] for i in one_class])) > 1:
            return False
    return True


def is_empty(one_class):
    return len(one_class[0]) <= 1


def most_common(one_class):
    lst = [i[0] for i in one_class]
    return max(set(lst), key=lst.count)


def confidence(one_class):
    most_common_class = most_common(one_class)
    return len([i[0] for i in one_class if i[0] == most_common_class]) / len(one_class)


def gain(one_class, attribute):
    d = [(entropy(i), len(i)) for i in split(one_class, attribute).values()]
    nAll = sum(i[1] for i in d)
    gain = sum([(i[0] * i[1]) / nAll for i in d])
    return gain


actual_classifier = "def classify(data):"


def build_tree(one_class, spaces="    "):
    global actual_classifier

    if is_empty(one_class) or is_pure(one_class):
        print(spaces, "then", most_common(one_class))
        print(spaces, "# confidence:", confidence(one_class))
        actual_classifier += "\n" + spaces + "return \"" + str(most_common(one_class)) + "\""
        return

    highest = get_highest_gain(one_class)
    d = split(one_class, highest)

    for key, value in d.items():
        print(spaces, "if", key)
        actual_classifier += "\n" + spaces + "if data[" + str(highest) + "] == \"" + str(key) + "\":"
        build_tree(value, spaces + "    ")


print("printing pseudo:")
build_tree(training_data)
print("\n\n")
print(actual_classifier)

exec(actual_classifier)
correct, wrong = 0, 0

for data in verification_data:
    if data[0] == classify(data):
        correct += 1
    else:
        wrong += 1

print("\n")
print("Correct classifications:", correct)
print("Wrong classifications:", wrong)
print("Accuracy: {0:.2f}%".format((correct / (correct + wrong) * 100)))
