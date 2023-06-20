import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings("ignore")

class TransForest:

    estimators_ = None
    feature_importances_ = None
    

    def __init__(self,
                 max_samples = 2,
                 criterion = 'entropy',
                 n_estimators= 100,
                 max_features = 10,
                 max_depth = 'log2'):
        self.criterion = criterion
        self.max_samples = max_samples
        self.max_depth = max_depth 
        self.n_estimators = n_estimators
        self.max_features = max_features


                    

    def fit(self, X, y):

        TransForest.estimators_ = []

        
        unlabeled = np.where(y == -1)[0]
        labeled = np.where(y != -1) [0]
        labeled_normal = np.where(y == 0)[0]
        labeled_anomaly = np.where(y == 1)[0]
        
        
        ratio = 0.1#labeled_anomaly.size / labeled.size 
        self.max_samples = max(256, self.max_samples * labeled.size)
        if self.max_depth == 'log2':
            self.max_depth = np.ceil(np.log2(self.max_samples))
        if self.max_features == 'sqrt':
            self.max_features = int(np.ceil(np.sqrt(X.shape[1])))

        
        unlabeled_sample_size = min(unlabeled.size, self.max_samples - labeled.size)

        TransForest.feature_importances_ = {k : 0 for k in range(X.shape[1])}
        
        for _ in range(self.n_estimators):

            if unlabeled.size == 0: # supervised case
                labeled_sample = np.random.randint(X.shape[0], size=self.max_samples)
                X_sample = X[labeled_sample, :]
                y_sample = y[labeled_sample]

                
            else: 
                # For each tree, injecting fixed labeled data + a subsample of the unlabeled data.
  
                unlabeled_sample = random.sample(list(unlabeled), unlabeled_sample_size)                              
                
                if (labeled.size) > 0: # semi-supervised case
                    sample_index = np.concatenate((labeled, unlabeled_sample))

                    X_sample = X[sample_index, :]
                    y_sample = y[sample_index] 
                else: # unsupervised case 
                    X_sample, y_sample = X[unlabeled_sample, :], y[unlabeled_sample]

            t = TransTree(self.max_depth, self.max_samples, self.max_features, self.criterion, TransForest.feature_importances_, ratio)
            t.fit(X_sample, y_sample)
            TransForest.estimators_.append(t)

        TransForest.feature_importances_ = self._get_feature_importances()

        return self


    def _path_length(self, X):

        avg_len = []

        for i in X:

            paths = []
            for t in TransForest.estimators_:
                
                node = t.root
                while node.left and node.right is not None:

                    if i[node.split_attr] < node.split_val:
                        node = node.left
                    else:
                        node = node.right


                if node.node_type == 'normal':

                    paths.append(self.max_depth +\
                                     2 * (np.log(self.max_samples - 1) + 0.5772156649) - \
                                     2 * (self.max_samples - 1) / self.max_samples)

                elif node.node_type == 'anomaly': # mark all points within the anomaly type node as anomalies.
                    paths.append(0)
                else:
                    paths.append(node.node_type * node.path_length)

            avg_len.append(sum(paths) / self.n_estimators)
            
        avg_len = np.array(avg_len).reshape(len(avg_len), 1)
        return avg_len


    def decision_function(self, X):


        avg_len = self._path_length(X)
        H = np.log(self.max_samples - 1) + 0.5772156649

        scores = []
        for length in avg_len:
            if length > 2:
                C = 2 * H - 2 * (self.max_samples - 1) / self.max_samples
            elif length == 2:
                C = 1
            else:
                C = 100000
            score =  2 ** -(length / C)
            scores.append(score)
            
        return np.array(scores)



    def _get_feature_importances(self):

        feature_weight = {}
        total_weight = sum([v for v in TransForest.feature_importances_.values()])
        for i in TransForest.feature_importances_.keys():
            feature_weight[i] = TransForest.feature_importances_[i] / total_weight # normalize
        return np.array([v for v in feature_weight.values()])


class TransTree:
    def __init__(self, max_depth, max_samples, max_features, criterion, feature_importances, ratio):
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.n_nodes = 1
        self.feature_importances = feature_importances
        self.max_samples = max_samples
        self.ratio = ratio



    def fit(self, X, y):

        self.root = self.fit_(X, y, 0)
        return self.root



      
    def _compute_impurity(self, count_normal, count_anomaly):

        if count_normal == 0 or count_anomaly == 0: 
            return 0

        p0 = count_normal / (count_normal + count_anomaly)
        p1 = count_anomaly / (count_normal + count_anomaly)

        if self.criterion == 'gini':
            impurity = 1 - p0 ** 2 - p1 ** 2    
        elif self.criterion == 'entropy':
            impurity = - (p0 * np.log2(p0) + p1 * np.log2(p1))
            
        return impurity
        



    def _compute_spread_gain(self, f, y, val):


        if len(np.unique(f)) <= 1:  # set gain = 0 if all values of the current dimension are the same.
            return 0
     
        num_bins = int(np.ceil(np.log2(f.shape[0]))) + 1
##        num_bins = int(np.ceil(np.sqrt(f.shape[0])))
        hist_total, bins = np.histogram(f, bins=num_bins)
        bins[bins == max(f)] = np.Inf
        bins[bins == min(f)] = np.NINF

        f_left = f[f < val]
        f_right = f[f >= val]

        # set the threshold for pseudolabelling


        sum_freq = np.sum(hist_total)
        thres = self.ratio * sum_freq

        ratio_normal_before, ratio_anomaly_before = 0, 0
        ratio_normal_after_l, ratio_anomaly_after_l = 0, 0
        ratio_normal_after_r, ratio_anomaly_after_r = 0, 0


        for i in range(num_bins): #loop through every bin 

            r0, r1 = 0, 0
            bound_min, bound_max = bins[i], bins[i + 1]
            pos = np.where((bound_min <= f) & (f < bound_max))[0]
            cur_data = f[pos]
            cur_label = y[pos]

            # Get label information of this bin
            count_labeled_normal = len(cur_label[cur_label == 0])
            count_labeled_anomaly = len(cur_label[cur_label == 1])

            # label spreading
            if count_labeled_normal + count_labeled_anomaly > 0: # if the bin contains labeled information

                if hist_total[i] >= thres and count_labeled_normal == 0: # if a dense bin contains only labeled anomalies
                    r0 = 1 - self.ratio
                    r1 = self.ratio

                else:# if the bin contains a mixture, spread the label based on the ratio between normal & anomlies

                    r0 = count_labeled_normal / (count_labeled_normal + count_labeled_anomaly)
                    r1 = count_labeled_anomaly / (count_labeled_normal + count_labeled_anomaly)

            else: # if no data is labeled, estimate the ratio based on the pre-defined thresholds
                if hist_total[i] >= thres:
                    r0 = 1 
                elif hist_total[i] < thres:
                    r1 = 1

            ratio_normal_before += r0 * hist_total[i]
            ratio_anomaly_before += r1 * hist_total[i]                    


            if bound_min <= val and val < bound_max : # locate the bin where the split occurs

                num_bin_left = len(cur_data[cur_data < val])
                num_bin_right = len(cur_data[cur_data >= val])

                try:
                    
                    ratio_normal_after_l += r0 * hist_total[i] * num_bin_left / (num_bin_left + num_bin_right)
                    ratio_anomaly_after_l += r1 * hist_total[i] * num_bin_left / (num_bin_left + num_bin_right)
                    ratio_normal_after_r += r0 * hist_total[i] * num_bin_right / (num_bin_left + num_bin_right)
                    ratio_anomaly_after_r += r1 * hist_total[i] * num_bin_right / (num_bin_left + num_bin_right)
                    
                except RuntimeWarning:
                    
                    pass


            elif bound_max <= val: # bins on the left child node
                ratio_normal_after_l += r0 * hist_total[i]
                ratio_anomaly_after_l += r1 * hist_total[i]
                

            else: # bins on the right child node
                ratio_normal_after_r += r0 * hist_total[i]
                ratio_anomaly_after_r += r1 * hist_total[i]
                        

        score_before = self._compute_impurity(ratio_normal_before, ratio_anomaly_before)
        score_after_l = self._compute_impurity(ratio_normal_after_l, ratio_anomaly_after_l)
        score_after_r = self._compute_impurity(ratio_normal_after_r, ratio_anomaly_after_r)

        gain = len(f) / self.max_samples * \
               (score_before - \
               (len(f_left) / (len(f_left) + len(f_right)) * score_after_l) - \
               (len(f_right) / (len(f_left) + len(f_right)) * score_after_r))

        return gain


    def fit_(self, X, y, height):
        
        count_normal = (y == 0).sum()
        count_anomaly = (y == 1).sum()

        # Leaf node:
        if height >= self.max_depth or len(X) <= 1:

            node_type = 1
            
            if count_normal > count_anomaly and count_normal == 1:
                node_type = 'normal'
            elif count_normal < count_anomaly: 
                node_type = 'anomaly'
            elif count_normal > 0 or count_anomaly > 0:
                node_type = count_normal / (count_anomaly + count_normal)

            size = len(X)
            if size > 2:
                height += 2 * (np.log(size - 1) + 0.5772156649) - \
                    2 * (size - 1) / size
            elif size == 2:
                height += 1


            return Node(None, None, None, None, y, node_type, height)


        init_gain = 0



        # Randomly select feature and splitting value
        best_attr = np.random.randint(0, X.shape[1])
        best_val = np.random.uniform(min(X[:, best_attr]), max(X[:, best_attr]))

        left_best = X[:, best_attr] < best_val                
        right_best = np.invert(left_best)

        X_left_best, X_right_best = X[left_best], X[right_best]
        y_left_best, y_right_best = y[left_best], y[right_best]


        for i in range(self.max_features): # finding best split
            
            split_attr = np.random.randint(0, X.shape[1])
            
            split_val = np.random.uniform(min(X[:, split_attr]), max(X[:, split_attr]))

            spread_gain = self._compute_spread_gain(X[:, split_attr], y, split_val)

            score_gain = spread_gain

            if score_gain > init_gain:
                init_gain = score_gain
                left_index = X[:, split_attr] < split_val                
                right_index = np.invert(left_index)
                X_left_best, X_right_best = X[left_index], X[right_index]
                y_left_best, y_right_best = y[left_index], y[right_index]
                best_attr, best_val = split_attr, split_val


        self.feature_importances[best_attr] += init_gain


        node = Node(best_attr, best_val, \
                    self.fit_(X_left_best, y_left_best, height+1), \
                    self.fit_(X_right_best, y_right_best, height+1))
        self.n_nodes += 2



        return node

  
class Node:
    def __init__(self, split_attr, split_val, left, right, label = None, node_type = None, path_length=None):
        self.split_attr = split_attr
        self.split_val = split_val
        self.left = left
        self.right = right
        self.path_length = path_length
        self.label = label
        self.node_type = node_type





