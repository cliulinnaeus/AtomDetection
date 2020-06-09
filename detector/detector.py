import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class detector():
    def __init__(self, img_size, x0, y0, dx, dy):
        self.img_size = img_size
        self.thresh = 0

        if x0+dx-1 > img_size or y0+dy-1 >img_size:
            raise Exception("Detection region is not in range")
        
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy




    def predict(self, data):
        labels_pred = []
        for d in data:
            box = d[self.x0:self.x0+self.dx, self.y0:self.y0+self.dy]
            tot_sum = np.sum(box)

            if tot_sum > self.thresh:
                labels_pred.append(1)
            else:
                labels_pred.append(0)
        return np.array(labels_pred)
        

    # take a validation set out of data
    # 
    def train(self, data, labels, verbose=False):
        data_tr, data_val, labels_tr, labels_val = train_test_split(data, labels, test_size=0.33)

        sums = []
        for d in data_tr:
            box = d[self.x0:self.x0+self.dx, self.y0:self.y0+self.dy]
            sums.append(np.sum(box))

        max_sum = np.max(sums)
        min_sum = np.min(sums)

        if verbose:
            plt.hist(sums, bins=50, normed=True)
            plt.show()

        thresh_candidates = np.linspace(min_sum, max_sum, int(max_sum-min_sum+1)//2)
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
    

    

