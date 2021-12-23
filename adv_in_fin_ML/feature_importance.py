
import sklearn
import pandas as pd
import sys
import datetime
from datetime import date, timedelta, datetime
import numpy as np
from feat_imp_test import featImpMDI, featImpMDA, featImpSFI

def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=10000):

    from sklearn.datasets import make_classification # make random dataset for classification

    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features,\
    n_informative=n_informative, n_redundant=n_redundant, random_state=0, shuffle=False)
    # trnsX = data, cont=label

    df0 = pd.date_range(end=datetime.today(), periods=n_samples, freq='B')


    trnsX = pd.DataFrame(trnsX, index=df0)
    cont = pd.Series(cont, index=df0).to_frame('bin')


    df0 = ['I_'+str(i) for i in range(n_informative)]+['R_'+str(i) for i in range(n_redundant)]
    df0 += ['N_'+str(i) for i in range(n_features-len(df0))]
    trnsX.columns = df0
    cont['w'] = 1./cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)

    return trnsX, cont

# 3 feature importance functions with substitution effects:

def featImpMDI(fit,featNames):
    # feat importance based on IS mean impurity reduction
    # only works with tree based classifiers
    df0={i:tree.feature_importances_ for i,tree
         in enumerate(fit.estimators_)}
    print(df0)
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan) # b/c max_features=1
    imp=(pd.concat({'mean':df0.mean(),
                    'std':df0.std()*df0.shape[0]**-0.5},
                   axis=1))
    imp/=imp['mean'].sum()
    return imp

def featImpMDA(clf,X,y,cv,sample_weight,t1,pctEmbargo,scoring='neg_log_loss'):
    # feat imporant based on OOS score reduction
    if scoring not in ['neg_log_loss','accuracy']:
        raise ValueError('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score
    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged cv
    scr0,scr1=pd.SEries(), pd.DataFrame(columns=X.columns)

    for i,(train,test) in enumerate(cvGen.split(X=X)):
        X0,y0,w0=X.iloc[train,:],y.iloc[train],sample_weight.iloc[train]
        X1,y1,w1=X.iloc[test,:],y.iloc[test],sample_weight.iloc[test]
        fit=clf.fit(X=X0,y=y0,sample_weight=w0.values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1)
            scr0.loc[i]=-log_loss(y1,prob,sample_weight=w1.values,
                                  labels=clf.classes_)
        else:
            pred=fit.predict(X1)
            scr0.loc[i]=accuracy_score(y1,pred,sample_weight=w1.values)

    for j in X.columns:
        X1_=X1.copy(deep=True)
        np.random.shuffle(X1_[j].values) # permutation of a single column
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1_)
            scr1.loc[i,j]=-log_loss(y1,prob,sample_weight=w1.values,
                                    labels=clf.classes_)
        else:
            pred=fit.predict(X1_)
            scr1.loc[i,j]=accuracy_score(y1,pred,sample_weight=w1.values)
    imp=(-scr1).add(scr0,axis=0)
    if scoring=='neg_log_loss':imp=imp/-scr1
    else: imp=imp/(1.-scr1)
    imp=(pd.concat({'mean':imp.mean(),
                    'std':imp.std()*imp.shape[0]**-0.5},
                   axis=1))
    return imp,scr0.mean()

def auxFeatImpSFI(featNames,clf,trnsX,cont,scoring,cvGen):
    imp=pd.DataFrame(columns=['mean','std'])
    for featName in featNames:
        df0=cvScore(clf,X=trnsX[[featName]],y=cont['bin'],
                    sample_weight=cont['w'],scoring=scoring,cvGen=cvGen)
        imp.loc[featName,'mean']=df0.mean()
        imp.loc[featName,'std']=df0.std()*df0.shape[0]**-0.5
    return imp

# generate random data and features
def getTestData(n_features=40, n_informative=10, n_redundant=10, n_samples=100):

    from sklearn.datasets import make_classification # make random dataset for classification
    trnsX, cont = make_classification(n_samples=n_samples, n_features=n_features,\
    n_informative=n_informative, n_redundant=n_redundant, random_state=0, shuffle=False)
    # trnsX = data, cont=label
    df0 = pd.date_range(end=datetime.today(), periods=n_samples, freq='B')
    trnsX = pd.DataFrame(trnsX, index=df0)
    cont = pd.Series(cont, index=df0).to_frame('bin')
    df0 = ['I_'+str(i) for i in range(n_informative)]+['R_'+str(i) for i in range(n_redundant)]
    df0 += ['N_'+str(i) for i in range(n_features-len(df0))]
    trnsX.columns = df0
    cont['w'] = 1./cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont

# snippet 7.3 p.109
from sklearn.model_selection._split import _BaseKFold
class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo

    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')

        # TODO: grouping function combinations insert here??
        # manage groups by using label in dataframe?
        # use combinations + group label to split into chunks??

        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[
            (i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),
                                                   self.n_splits)
        ]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxT1Idx<X.shape[0]: # right train ( with embargo)
                train_indices=np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
            yield train_indices,test_indices


# snippet 7.4 p.110
def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',
            t1=None,cv=None,cvGen=None,pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    idx = pd.IndexSlice
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[idx[train],:],y=y.iloc[idx[train]],
                    sample_weight=sample_weight.iloc[idx[train]].values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[idx[test],:])
            score_=-log_loss(y.iloc[idx[test]], prob,
                             sample_weight=sample_weight.iloc[idx[test]].values,
                             labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[idx[test],:])
            score_=accuracy_score(y.iloc[idx[test]],pred,
                                  sample_weight=sample_weight.iloc[idx[test]].values)
        score.append(score_)
    return np.array(score)




def featImportances(trnsX,cont,n_estimators=1000,cv=10,
                    max_samples=1.,numThreads=11,pctEmbargo=0,
                    scoring='accuracy',method='SFI',minWLeaf=0.,**kargs):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import cross_val_score
    #from mpEngine import mpPandasObj
    n_jobs=(-1 if numThreads>1 else 1) # run 1 thread w/ ht_helper in dirac1
    #1) prepare classifier,cv. max_features=1, to prevent masking
    clf=DecisionTreeClassifier(criterion='entropy',max_features=1,
                               class_weight='balanced',
                               min_weight_fraction_leaf=minWLeaf)

    clf=BaggingClassifier(base_estimator=clf,n_estimators=n_estimators,
                          max_features=1.,max_samples=max_samples,
                          oob_score=True,n_jobs=n_jobs)

    fit=clf.fit(X=trnsX,y=cont['bin'],sample_weight=cont['w'].values)
    oob=fit.oob_score_
    if method=='MDI':
        imp=featImpMDI(fit,featNames=trnsX.columns)
        oos=cvScore(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],
                    t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring).mean()
    elif method=='MDA':
        imp,oos=featImpMDA(clf,X=trnsX,y=cont['bin'],cv=cv,
                           sample_weight=cont['w'],t1=cont['t1'],
                           pctEmbargo=pctEmbargo,scoring=scoring)
    elif method=='SFI':
        cvGen=PurgedKFold(n_splits=cv,t1=cont['t1'],pctEmbargo=pctEmbargo)
        oos=cvScore(clf,X=trnsX,y=cont['bin'],sample_weight=cont['w'],
                    scoring=scoring,cvGen=cvGen).mean()
        clf.n_jobs=1 # parallelize auxFeatImpSFI rather than clf
        imp=pmPandasObj(auxFeatImpSFI,('featNames',trnsX.columns),numThreads,
                        clf=clf,trnsX=trnsX,cont=cont,scoring=scoring,cvGen=cvGen)
    return imp,oob,oos

if name == '__main__':
    # to use it in another file: sys.path.insert(1, 'path/to/this/folder')
    # and import feature_importance
    # only need to use featImportances function
    X, y = getTestData()
    imp,oob,oos = featImportances(X,y,method='MDI')
    print('imp:\n',imp,'\n')
    print('oob:\n',oob,'\n')
    print('oos:\n',oos,'\n')
    
    
