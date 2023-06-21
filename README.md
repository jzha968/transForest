# TransForest
This is a simple implementation of the paper 'A Transductive Forest for Anomaly Detection with Few Labels'. 

Given very limited label information, TransForest offers:

1. Semi-supervised anomaly score learned from both labeled and unlabeled data.
2. Feature importance ranking consistent with the rankings provided by popular supervised forests (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html, https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) on low-dimensional data sets . 
## Example 
Following is an example of using TransForest. Script to reproduce more experiment result in the paper is available in ``run``.
```python
from transforest import TransForest

transForest = TransForest()
transForest.fit(X_train, y_train)
# Compute anomaly score
anomaly_score = transForest.decision_function(X_test)
# Compute feature importance
feature_importance = transForest.feature_importances_
```
## Package denpendencies
- python == 3.10.5
- sklearn == 1.1.1
- numpy == 1.23.1
- pyod == 1.0.5
- ADBench
- HIF
