

############################3
# this code takes FOREVER to run, need to figure out why
############################


from __future__ import print_function
from __future__ import division
import time
start = time.time()

from utils import (
    save_model,
    load_model,
    neutralize,
    get_biggest_change_features,
    validation_metrics,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
    EXAMPLE_PREDS_COL
)
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import pandas as pd
    from joblib import parallel_backend
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score, cross_val_predict
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import SVR, SVC
    import matplotlib.pyplot as plt
    import scipy.interpolate
    from itertools import product, chain
    from sklearn.metrics import make_scorer, mean_squared_error
    from bayes_opt import BayesianOptimization
import pickle
from halo import Halo
import json
import sys

#wandb.init(project='numerai1',entity='hl5817')
spinner = Halo(text='', spinner='dots')
# xgboost + bayesian opt:
# https://www.kaggle.com/tilii7/xgboost-bayesian-optimization/script
# reduce runtime using this paper?
# https://arxiv.org/pdf/1710.11547v1.pdf

###########################################################
#   I. select features that will be used for prediction
###########################################################


with open("features.json", "r") as f:
    feature_metadata = json.load(f)
features = feature_metadata["feature_sets"]["small"] # names of features
# feature_metadata['feature_sets']['small'] -> 38 features
# medium => 420
read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
training_data = pd.read_parquet('training_data.parquet',columns=read_columns)

validation_data = pd.read_parquet('validation_data.parquet',columns=read_columns)


print('train shape: ',training_data.shape)
print('validation shape: ',validation_data.shape)


#####################################################
# II. create XGBoost model for regression
#####################################################

# input data for training
dtrain = xgb.DMatrix(training_data[features],label=training_data[TARGET_COL])
# test data
dval = xgb.DMatrix(validation_data[features])
y_val = validation_data[TARGET_COL]

RMSEbest = 10.
ITERbest = 0
def XGbcv(max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):

    global RMSEbest
    global ITERbest

    paramt = {
              'booster' : 'gbtree',
              'max_depth' : max_depth.astype(int),
              'gamma' : gamma,
              'eta' : 0.01,
              'objective': 'reg:squarederror',# used to be reg:linear
              'nthread' : 8,
              #'silent' : True,
              'eval_metric': 'rmse',
              'subsample' : subsample,
              'colsample_bytree' : colsample_bytree,
              'min_child_weight' : min_child_weight,
              'max_delta_step' : max_delta_step.astype(int),
              'seed' : 1001
              }

    folds = 5

    xgbr = xgb.cv(
           paramt,
           dtrain,
           num_boost_round = 100, # same thing as n_estimators, greatly impacts optimization time
#           stratified = True,
           nfold = folds,
           verbose_eval = False,
           early_stopping_rounds = 50,
           metrics = "rmse",
           show_stdv = True
          )

    cv_score = xgbr['test-rmse-mean'].iloc[-1]
    if ( cv_score < RMSEbest ):
        RMSEbest = cv_score
        ITERbest = len(xgbr)

    return (-1.0 * cv_score)

#########################################################
# III. create bayesian optimization process for the model
########################################################
print('optimizing...')
XGbBO = BayesianOptimization(XGbcv, {'max_depth': (3, 10),
                                     'gamma': (0.00001, 1.0),
                                     'min_child_weight': (0, 5),
                                     'max_delta_step': (0, 5),
                                     'subsample': (0.5, 0.9),
                                     'colsample_bytree' :(0.05, 0.4)
                                    })

with parallel_backend('multiprocessing'):
    XGbBO.maximize(init_points=10, n_iter=1, acq="ei", xi=0.01)
# init points = how many steps of random exploration you want
# n_iter = how many steps of BO you want

###########################################################
# IV. get optimal parameters from BO and train model with it
###########################################################
print('done')
opt_params = XGbBO.max['params']
print('opt_params: ',opt_params)

best_RMSE = XGbBO.max['target']
max_depth = opt_params['max_depth']
gamma = opt_params['gamma']
min_child_weight = opt_params['min_child_weight']
max_delta_step = opt_params['max_delta_step']
subsample = opt_params['subsample']
colsample_bytree = opt_params['colsample_bytree']

paramt = {'booster' : 'gbtree', 'max_depth' : max_depth.astype(int), 'gamma' : gamma, 'eta' : 0.01, 'objective': 'reg:squarederror', 'nthread' : 8,\
              'eval_metric': 'rmse', 'subsample' : subsample, 'colsample_bytree' : colsample_bytree, 'min_child_weight' : min_child_weight,\
              'max_delta_step' : max_delta_step.astype(int), 'seed' : 1001}

folds = 5
spinner.start('training model w/ optimal params...')
xgbr = xgb.train(paramt, dtrain, num_boost_round=int(ITERbest*(1+(1/folds))))
spinner.succeed()

######################################################
# V. predict on validation data w/ optimal parameters
######################################################
spinner.start('predicting on validation data w/ optimal params...')
pred = xgbr.predict(dval)
mse = ((y_val - pred)**2).mean(axis=0)
spinner.succeed()
print('model mse is: ', mse)

spinner.start('saving trained model...')
# save model after training
try:
    pickle.dump(xgbr, open('./models/bayesopt_xgbr.pkl','wb'))
except:
    pickle.dump(xgbr, open('bayesopt_xgbr.pkl','wb'))
spinner.succeed()
end = time.time()
print('code took ',(end-start)/60,'min to run')
