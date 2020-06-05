import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class detector():
    def __init__(self, img_size):
        self.img_size = img_size
        self.thresh = 0

    def predict(self, data):
        labels_pred = []
        for d in data:
            tot_sum = np.sum(d)

            if tot_sum > self.thresh:
                labels_pred.append(1)
            else:
                labels_pred.append(0)
        return np.array(labels_pred)
        

    # take a validation set out of data
    # 
    def train(self, data, labels, verbose=False):
        data_tr, data_val, labels_tr, labels_val = train_test_split(data, labels, test_size=0.3)

        sums = []
        for d in data_tr:
            sums.append(np.sum(d))

        max_sum = np.max(sums)
        min_sum = np.min(sums)

        if verbose:
            plt.hist(sums, bins=30)
            plt.show()

        thresh_candidates = np.linspace(min_sum, max_sum, int(max_sum-min_sum+1))
        accuracies = []
        for thresh in thresh_candidates:
            self.thresh = thresh
            labels_pred = self.predict(data_val)
            accuracies.append(accuracy_score(labels_val, labels_pred))
        idx = np.argmax(accuracies)
        best_thresh = thresh_candidates[idx]
        self.thresh = best_thresh
        
        y_hat = self.predict(data_tr)
        accuracy_tr = accuracy_score(labels_tr, y_hat)

        print(f"Training accuracy: {accuracy_tr}")
        print(f"Validation accuracy: {np.max(accuracies)}")
        print(f"Threshold: {self.thresh}")

        return accuracy_tr, np.max(accuracies), self.thresh
    

    

