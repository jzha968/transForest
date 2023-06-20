from __future__ import absolute_import
import numpy as np
import os
import random
from sklearn.metrics import roc_auc_score
from transforest import TransForest
from hif import hiForest
from baseline.DeepSAD.src.run import DeepSAD
from baseline.DevNet.run import DevNet
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.xgbod import XGBOD
from pyod.models.ocsvm import OCSVM


def data_generator(d, p, r):
    
    '''
    Generate partially labeled data.
    p = label percentage.
    r = |labeled normal point|/|labeled anomaly| ratio.
    '''

    data = np.load(f_path + '/' + d)
    X, y = data["X"], data["y"].ravel()

    anomaly_index = np.where(y == 1)[0]
    normal_index = np.where(y == 0)[0]
    
    num_labeled_anomaly = int(np.ceil(p * len(anomaly_index)))
    num_labeled_normal = min(num_labeled_anomaly * r, normal_index.size)

    labeled_anomaly_index = random.sample(list(anomaly_index), num_labeled_anomaly)
    labeled_normal_index = random.sample(list(normal_index), num_labeled_normal)

    labeled_index = labeled_anomaly_index + labeled_normal_index
    unlabeled_index = [i for i in range(X.shape[0]) if i not in labeled_index]
        
    y_partial_train = []
    for i in range(y.shape[0]):
        if i in labeled_anomaly_index:
            y_partial_train.append(1)
        elif i in labeled_normal_index:
            y_partial_train.append(0)
        else:
            y_partial_train.append(-1)

    y_partial_train = np.array(y_partial_train).reshape(-1, 1)
    X_labeled = X[labeled_index, :]
    y_labeled = np.ravel(y[labeled_index])

    return X, y, y_partial_train, X_labeled, y_labeled

def compute_AUC(d):
    
    interval_1 = [i / 100 for i in range(1, 21)]
    interval_2 = [i * 5 / 100 for i in range(5, 11)]
    num_labeled = interval_1 + interval_2
    num_labeled = [0.1]

    res_dict = {'transForest': [],
                'hybridiForest':[],
                'XGBOD': [],
                'DevNet': [],
                'DeepSAD': []}
 
    for i in num_labeled:
        
        raw_X, raw_y, partial_y, labeled_X, labeled_y = data_generator(d, i, 1)
        test_index = np.where(partial_y == -1)[0]
        X_test, y_test = raw_X[test_index, :], raw_y[test_index]

        ### TransForest ###
        transForest = TransForest().fit(raw_X, partial_y)
        score_transForest = transForest.decision_function(X_test)
        res_dict['transForest'].append(roc_auc_score(y_test, score_transForest))

        ### Hybrid iForest ###
        hif = hiForest(raw_X, 100, 256)
        score_hif = []
        ''' add supervised information to hif '''
        alpha_1, alpha_2 = 0.2, 0.7
        for i in range(raw_X.shape[0]):
            if int(partial_y[i]) == 1:
                hif.addAnomaly(np.ravel(raw_X[i, :]), 1)
        ''' aggregate scores '''
        for x in X_test:
            scores = hif.computeAggScore(x)
            S, Sc, Sa = scores[0], scores[2], scores[3]
            min_Scores = min([S, Sc, Sa])
            max_Scores = max([S, Sc, Sa])
            norm_S = (S - min_Scores) / (max_Scores - min_Scores)
            norm_Sc = (Sc - min_Scores) / (max_Scores - min_Scores)
            norm_Sa = (Sa - min_Scores) / (max_Scores - min_Scores)
            Sx = alpha_2 * ( alpha_1 * norm_S + (1 - alpha_1) * norm_Sc) + (1 - alpha_2) * norm_Sa
            score_hif.append(Sx)
        res_dict['hybridiForest'].append(roc_auc_score(y_test, score_hif))

        ### XGBOD ###

        xgbod = XGBOD().fit(labeled_X, labeled_y)
        score_xgbod = xgbod.decision_function(X_test)
        res_dict['XGBOD'].append(roc_auc_score(y_test, score_xgbod))

        ### DevNet ###
        devnet = DevNet(seed = 42).fit(raw_X, partial_y)
        score_devnet = devnet.predict_score(X_test)
        res_dict['DevNet'].append(roc_auc_score(y_test, score_devnet))

        ### DeepSAD ###
        partial_y[partial_y == 0] = -1
        deepsad = DeepSAD(seed = 42).fit(raw_X, partial_y)
        score_deepsad = deepsad.predict_score(X_test)
        res_dict['DeepSAD'].append(roc_auc_score(y_test, score_deepsad))

    return res_dict


def compute_feature_importances(d):

    num_labeled = 0.1
    raw_X, raw_y, partial_y, labeled_X, labeled_y = data_generator(d, num_labeled, 1)

    res_dict = {'transForest': None,
                'extraTrees': None,
                'randomForest': None}
    
    ### TransForest ###
    transForest = TransForest().fit(raw_X, partial_y)
    res_dict['transForest'] = transForest.feature_importances_

    ### ExtraTrees ###
    extraTrees = ExtraTreeClassifier().fit(raw_X, raw_y)
    res_dict['extraTrees'] = extraTrees.feature_importances_

    ### RandomForest ###
    randomForest = RandomForestClassifier().fit(raw_X, raw_y)
    res_dict['randomForest'] = randomForest.feature_importances_

    for k, v in res_dict.items():
        res_dict[k] = np.argsort(-1 * v)

    return res_dict

def main(datasets):

    ### AUC ###
    num_trial = 1
    auc_dict = {'transForest': [],
                'hybridiForest':[],
                'XGBOD': [],
                'DevNet': [],
                'DeepSAD': []}

    for i in range(num_trial):
##        seed = random.seed(42)
        res = compute_AUC(datasets)
        for k, v in res.items():
            auc_dict[k].append(v)
            
    for k, v in res.items():
        auc_dict[k] = (np.mean(v, axis = 0),
                       np.std(v, axis = 0))

    return auc_dict

##    ### Feature importance ranking ###
##    feature_ranking = compute_feature_importances(datasets)
##    return feature_ranking
        
    
f_path = './datasets/Classical'
##datasets = os.listdir(f_path)
datasets = ['1_ALOI.npz', '20_letter.npz', '23_mammography.npz', '24_mnist.npz',
            '26_optdigits.npz', '28_pendigits.npz', '29_Pima.npz', '2_annthyroid.npz',
            '30_satellite.npz', '31_satimage-2.npz', '32_shuttle.npz',
            '36_speech.npz', '38_thyroid.npz', '4_breastw.npz', '6_cardio.npz']


for d in datasets:
    print(d)
    print(main(d))
