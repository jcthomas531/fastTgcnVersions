from sklearn.metrics import make_scorer
import numpy as np
#point based iou score for model performance

def iouPoint(observed, predicted):
    pred1 = predicted == 1
    obs1 = observed == 1
    numer = np.sum(pred1 & obs1)
    denom = np.sum(pred1 | obs1)
    return numer/denom

iouScorer = make_scorer(iouPoint)