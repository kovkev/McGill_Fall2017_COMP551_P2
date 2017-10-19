

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


if __name__ == '__main__':
    import sys
    from naivebayes import NaiveBayes
    xfilename = sys.argv[1]
    yfilename = sys.argv[2]
    testset = sys.argv[3]

    print("training naive bayes...")
    n = NaiveBayes()
    n.train_set(convert(xfilename, yfilename))
    predictions = []

    print("creating predictions...")
    with open(testset, "r") as testfile:
        data = testfile.read()
        lines = data.split('\n')[1:]
        for line in lines:
            if not line:
                continue
            features = line.split(",")[1].split(" ")
            predictions.append(n.predict(features))

    print("outputing predictions...")
    output_prediction(predictions)
