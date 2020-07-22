import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class detector():
    def __init__(self, img_size, x0, y0, dx, dy):
        self.img_size = img_size
        self.thresh = 0

        if x0+dx-1 > img_size or y0+dy-1 >img_size:
            raise Exception("Detection region is not in range")
        
        self.x0 = x0 - dx//2
        self.y0 = y0 - dy//2
        self.dx = dx
        self.dy = dy




    def predict(self, data):
        labels_pred = []
        # if data is just a matrix
        if len(data.shape) == 2:
            data = np.array([data])
        for d in data:
            box = d[self.y0:self.y0+self.dy, self.x0:self.x0+self.dx]
            tot_sum = np.sum(box)
            # print(tot_sum)
            if tot_sum > self.thresh:
                labels_pred.append(1)
            else:
                labels_pred.append(0)
        # if show_atom_sig:

        return np.array(labels_pred)
        

    # take a validation set out of data
    # 
    def train(self, data, labels, val_size=0.33, verbose=False):
        data_tr, data_val, labels_tr, labels_val = train_test_split(data, labels, test_size=val_size)

        sums = []
        for d in data_tr:
            box = d[self.y0:self.y0+self.dy, self.x0:self.x0+self.dx]
            sums.append(np.sum(box))

        max_sum = np.max(sums)
        min_sum = np.min(sums)

        # print(f"maxsum:{max_sum}")
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

        if verbose:
            print(f"Training accuracy: {accuracy_tr}")
            print(f"Validation accuracy: {np.max(accuracies)}")
            print(f"Threshold: {self.thresh}")

        return accuracy_tr, np.max(accuracies), self.thresh
    
    def visualize_data(self, data, figsize=5):
        fig = plt.figure(figsize=(figsize, figsize))

        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.imshow(data, cmap='hot')
        ax.set_aspect('equal')
        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        cbaxes = fig.add_axes([0.95, 0.1, 0.03, 0.8]) 
        plt.colorbar(orientation='vertical', cax=cbaxes)
        plt.xlabel("x [pixels]")
        plt.ylabel("y [pixels]")
        # cb = plt.colorbar(ax, cax = cax)  
        rect = patches.Rectangle((self.x0, self.y0), self.dx, self.dy, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        plt.show()
    

    #TODO: def roc_curve(true_labels, scores)
            # plots roc curve
            # returns fpr, tpr, auc 



