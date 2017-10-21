# Why we chose adaboost
# It helps reduce bias
# Since Naive Bayes is considered to be high bias/low variance


import math
from naivebayes import NaiveBayes

class AdaBoost(object):

  def __init__(self, laplace=1):
    self.model = NaiveBayes

  def train_set(self, dataset, M=20):
    N = len(dataset)
    ws = [1.0 for i in dataset]
    models = [self.model() for i in range(M)]
    errs = [1.0 for i in range(M)]
    alphas = [1.0 for i in range(M)]

    # M = 1
    for m in range(M):
      print("m is", m)
      for n in range(N):
        example = dataset[n]
        # print("n is")
        # print(ws[n])
        models[m].train(*example, ws[n])

      incorrects = [1 if models[m].predict(*dataset[n][1:]) != dataset[n][0] else 0 for n in range(N)]
      numerator = sum([ws[n] * incorrects[n] for n in range(N)])
      denominator = float(sum(ws))

      errs[m] = numerator / denominator

      alphas[m] = math.log( (1 - errs[m]) / errs[m])

      incorrects = [1 if models[m].predict(*dataset[n][1:]) != dataset[n][0] else 0 for n in range(N)]
      ws = [ws[n] * math.exp( alphas[m] * incorrects[n]) for n in range(N)]

    self.models = models
    self.alphas = alphas

  def predict(self, features):
    weights = {}
    M = len(self.models)
    for m in range(M):
      prediction = weights[self.models[m].predict(features)]

      if prediction not in weights:
        weights[prediction] = 0.0

      weights[prediction] += self.alphas[m]

    max_prediction = (None, 0)
    for k,v in weights.items():
      if v > max_prediction[1]:
        max_prediction = (k,v)

    return max_prediction[1]


