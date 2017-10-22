from mpmath import mpf, mpc, mp

import time
current_milli_time = lambda: int(round(time.time() * 1000))

mp.dps = 40

class NaiveBayes(object):

    def __init__(self, laplace=1):
        self._frequencies = []  # counts frequencies
        self._classes = {}  # stores class to index
        self._features = {}  # stores features to index
        self._feature_counts = []  # stores probabilities for features
        self._feature_probabilies = {}
        self._class_counts = []  # stores probabilites for class
        self.entries = 0
        self.laplace = laplace
        self._dirty = True
        self.prediction_count = 0


    def predict(self, features):
        self.prediction_count += 1
        if self.prediction_count % 10000 == 0:
            print(self.prediction_count)

        # Create the prediction
        if self._dirty:
            self._calculate_feature_probabilities()
            self._dirty = False
        predicted_class = None
        predicted_max = 0
        for class_, cindex in self._classes.items():
            class_probability = self._class_counts[cindex] / self.entries
            given_class_probability = 1
            feature_probability = mpf(1.0)
            for feature in features:
                # Ignore any feature we've never seen before
                if feature not in self._features:
                    continue
                findex = self._features[feature]
                feature_probability *= self._feature_probabilies[findex]
                # Perform laplace filtering
                given_class = \
                    (self.laplace + self._frequencies[cindex][findex]) \
                    / (self._class_counts[cindex] + self.laplace
                        * len(self._features))
                given_class_probability *= given_class
            probability = given_class_probability * class_probability \
                / feature_probability
            if probability > predicted_max:
                predicted_class = class_
                predicted_max = probability
        return predicted_class

    def train(self, class_, features, n=1.0):
        self._dirty = True
        self.entries += 1.0 * n
        if class_ not in self._classes:
            self._classes[class_] = len(self._classes)
            self._frequencies.append([0.0 for i in range(len(self._features))])
            self._class_counts.append(0.0)
        for feature in features:
            if feature not in self._features:
                self._features[feature] = len(self._features)
                self._feature_counts.append(0)
                for features_given_class_counts in self._frequencies:
                    features_given_class_counts.append(0)
        class_index = self._classes[class_]
        self._class_counts[class_index] += 1.0 * n
        for feature in features:
            feature_index = self._features[feature]
            self._feature_counts[feature_index] += 1.0 * n
            self._frequencies[class_index][feature_index] += 1.0 * n

    def train_set(self, training_set):
        for entry in training_set:
            self.train(*entry)

    def _calculate_feature_probabilities(self):
        self._feature_probabilies = [0.0 for f in self._features]
        feature_count = sum(self._feature_counts)
        for feature, findex in self._features.items():
            for class_, cindex in self._classes.items():
                self._feature_probabilies[findex] += \
                    self._frequencies[cindex][findex]
            self._feature_probabilies[findex] /= feature_count


if __name__ == '__main__':
    n = NaiveBayes()
    n.train("a", ["a", "b"])
    n.train("a", ["a", "b"])
    n.train("b", ["c", "d"])
    n.train("a", ["a", "b"])
    print(n.predict(["c", "a", "b", "a"]))
