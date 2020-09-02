import numpy as np

class RW:
    def __init__(self):
        pass

    def predict(self, X_test):
        predicted = []
        for row in X_test:
            predicted.append([row[-1]])

        predicted = np.array(predicted)

        return predicted

