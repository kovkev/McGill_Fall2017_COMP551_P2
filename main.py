import time
current_milli_time = lambda: int(round(time.time() * 1000))


def convert(xfilename, yfilename):
    """Reads in xfile and yfile and converts the data to something naivebayes
    can understand."""
    xinputs = {}
    with open(xfilename, "r") as xfile:
        data = xfile.read()
        lines = data.split('\n')[1:]
        for line in lines:
            if not line:
                continue
            line = line.split(",")
            xinputs[int(line[0])] = line[1]
    yinputs = {}
    with open(yfilename, "r") as yfile:
        data = yfile.read()
        lines = data.split('\n')[1:]
        for line in lines:
            if not line:
                continue
            line = line.split(",")
            yinputs[int(line[0])] = int(line[1])
    converted_values = []
    for index in xinputs:
        features = list("".join(xinputs[index].split()))
        converted_values.append((yinputs[index], features))
    return converted_values


def output_prediction(predictions, filename="baseline.csv"):
    """Assuming index of prediction is the index in predictions, will output
    a csv file in the correct format"""
    strings = []
    for i, prediction in enumerate(predictions):
        strings.append(str(i) + "," + str(prediction))
    with open(filename, "w") as file:
        file.write("Id,Category\n")
        file.write("\n".join(strings))

def split_train_validation(dataset, num_validation):
    train = dataset[:]
    valid = []
    from random import randint
    for i in range(num_validation):
        index = randint(0, len(train)-1)
        valid.append(train.pop(index))

    return train, valid

def time_before(s, before=True):
    print("time %s %s : %s" % ("before" if before else "after", s, current_milli_time()))

def time_after(s):
    time_before(s, False)


if __name__ == '__main__':
    import sys
    from adaboost import AdaBoost
    from naivebayes import NaiveBayes
    xfilename = sys.argv[1]
    yfilename = sys.argv[2]
    testset = sys.argv[3]

    print("training naive bayes...")
    nb = NaiveBayes()
    ab = AdaBoost()
    dataset = convert(xfilename, yfilename)
    validation_set_size = 10000
    train_set, validation_set = split_train_validation(dataset, validation_set_size)
    num_to_train_on = 10000000
    time_before("training adaboost")
    ab.train_set(dataset[:num_to_train_on])
    time_after("training adaboost")
    time_before("training naive bayes")
    nb.train_set(dataset[:num_to_train_on])
    time_after("training naive bayes")

    kg_validations_nb = []
    kg_validations_ab = []

    for i in validation_set:
        kg_validations_nb.append(nb.predict(*i[1:]) == i[0])
        kg_validations_ab.append(ab.predict(*i[1:]) == i[0])

    # print("Errors nb: %s " % sum([0 if i else 1 for i in kg_validations_nb]))
    print("Errors ab: %s " % sum([0 if i else 1 for i in kg_validations_ab]))

    # import pdb; pdb.set_trace()

    predictions = []

    print("creating predictions...")
    with open(testset, "r") as testfile:
        data = testfile.read()
        lines = data.split('\n')[1:][:num_to_train_on]
        for line in lines:
            if not line:
                continue
            features = line.split(",")[1].split(" ")
            predictions.append(ab.predict(features))

    print("outputing predictions...")
    output_prediction(predictions)
