import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
from sklearn.model_selection import GridSearchCV
import matplotlib.pylab as plt
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 16, 4


def xgb_tune_trees_num(alg,
                       dtrain,
                       target,
                       useTrainCV=True,
                       cv_folds=3,
                       early_stopping_rounds=50,
                       seed=42,
                       verbose=True,
                       scoring='neg_log_loss',
                       plot_feature_importance=False):
    xgb_score = scoring
    if xgb_score == 'neg_log_loss':
        xgb_score = 'logloss'
    elif xgb_score == 'roc_auc':
        xgb_score = 'auc'
    else:
        raise 'unknown metric'
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=target.ravel())
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics=xgb_score, early_stopping_rounds=early_stopping_rounds, as_pandas=True, seed=seed)
        alg.set_params(n_estimators=cvresult.shape[0])
        if verbose:
            print(cvresult.tail(1))
    alg.fit(dtrain, target.ravel(), eval_metric=xgb_score)
    dtrain_predprob = alg.predict_proba(dtrain)[:, 1]
    score = 0.0
    if scoring == 'neg_log_loss':
        score = metrics.log_loss(target.ravel(), dtrain_predprob)
    elif scoring == 'roc_auc':
        score = metrics.roc_auc_score(target.ravel(), dtrain_predprob)
    if verbose:
        print("\nModel Report")
        print("Score (Train): %f" % score)
    if plot_feature_importance:
        feat_imp = pd.Series(alg.booster().get_score(importance_type='gain')).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
    result = {}
    if useTrainCV:
        result['n_estimators'] = alg.get_params()['n_estimators']
    result['score_train'] = score
    return result


def xgb_tune(algorithm=None,
             X=np.array([[0]]),
             y=np.array([0]),
             param_dict=None,
             early_stopping_rounds=50,
             nfolds=3,
             randomstate=42,
             verbose=True,
             cv_thread_num=1,
             scoring='neg_log_loss'):
    '''tuneable parameters:
       'max_depth',
       'min_child_weight',
       'gamma',
       'subsample',
       'colsample_bytree',
       'reg_alpha',
       'reg_lambda',
       'learning_rate' '''
    if param_dict is None:
        param_dict = {
            'max_depth': [2, 4, 6, 8, 10],
            'min_child_weight': [1, 3, 5, 10, 50],
            'gamma': [0, 0.01, 0.1, 0.15, 0.2],
            'subsample': [0.4, 0.6, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.4, 0.6, 0.8, 0.9, 1.0],
            'reg_alpha': [0.01, 0.1, 1, 10, 100],
            'reg_lambda': [0.01, 0.1, 1, 10, 100],
            'learning_rate': [0.001, 0.008, 0.05, 0.1, 0.5]
        }
    step_batches = {
        2:{'max_depth':param_dict['max_depth'],
           'min_child_weight':param_dict['min_child_weight']},
        3:{'gamma':param_dict['gamma']},
        4:{'subsample':param_dict['subsample'],
           'colsample_bytree':param_dict['colsample_bytree']},
        5:{'reg_alpha':param_dict['reg_alpha'],
           'reg_lambda':param_dict['reg_lambda']},
        6:{'learning_rate':param_dict['learning_rate']}
    }
    if algorithm is None:
        algorithm = XGBClassifier(learning_rate=0.1,
                         n_estimators=1000,
                         max_depth=5,
                         min_child_weight=1,
                         gamma=0,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective='binary:logistic',
                         nthread=1,
                         scale_pos_weight=1,
                         seed=randomstate)
    else:
        algorithm.nthread = 1
    xgb_tune_trees_num(alg=algorithm,
                       dtrain=X,
                       target=y,
                       useTrainCV=True,
                       cv_folds=nfolds,
                       early_stopping_rounds=early_stopping_rounds,
                       seed=randomstate,
                       verbose=verbose,
                       scoring=scoring)
    best_estimator = algorithm
    for step in range(2,7):
        gsearch = GridSearchCV(estimator=best_estimator,
                                param_grid=step_batches[step],
                                scoring=scoring,
                                n_jobs=cv_thread_num,
                                iid=False,
                                cv=KFold(n_splits=nfolds, shuffle=True, random_state=randomstate))
        gsearch.fit(X, y.ravel())
        if verbose:
            print('step%i:'%step, gsearch.best_params_, 'best score:', abs(gsearch.best_score_))
        xgb_tune_trees_num(alg=gsearch.best_estimator_,
                           dtrain=X,
                           target=y,
                           useTrainCV=True,
                           cv_folds=nfolds,
                           early_stopping_rounds=early_stopping_rounds,
                           seed=randomstate,
                           verbose=verbose,
                           scoring=scoring)
        best_estimator = gsearch.best_estimator_
    if verbose:
        print(best_estimator.get_xgb_params())
    return best_estimator

