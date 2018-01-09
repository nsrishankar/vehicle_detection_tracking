# Functions that comprise of all decided classifiers, as well as some deep learning algorithms

import numpy as np
import cv2
import pickle
import time

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score
#from keras.wrappers.scikit_learn import KerasClassifier

#from neural_networks import LeNet5

def linear_svm(X_train,y_train,X_test,y_test):
    t0_classifier=time.time()
    clf=LinearSVC(penalty='l2')
    clf.fit(X_train,y_train)
    t1_classifier=time.time()
    
    accuracy=round(100*clf.score(X_test,y_test),4)
    train_time=round(t1_classifier-t0_classifier,4)
    
    save_model='linear_svm_classifier.p'
    pickle.dump(clf,open(save_model,'wb'))
    
    return clf, accuracy, train_time

def svm(X_train,y_train,X_test,y_test):
    t0_classifier=time.time()
    clf=SVC(tol=1e-3, kernel='polynomial')
    clf.fit(X_train,y_train)
    t1_classifier=time.time()
    
    accuracy=round(100*clf.score(X_test,y_test),4)
    train_time=round(t1_classifier-t0_classifier,4)
    
    save_model='svm_classifier.p'
    pickle.dump(clf,open(save_model,'wb'))
    
    return clf, accuracy, train_time

def Adaboost(X_train,y_train,X_test,y_test):
    t0_classifier=time.time()
    clf=AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    t1_classifier=time.time()

    accuracy=round(100*clf.score(X_test,y_test),4)
    train_time=round(t1_classifier-t0_classifier,4)
    
    save_model='adaboost_classifier.p'
    pickle.dump(clf,open(save_model,'wb'))
    
    return clf, accuracy, train_time 

def ensemble_ml(X_train,y_train,X_test,y_test):
    t0_classifier=time.time()
    clf_1=LinearSVC(penalty='l2')
    clf_2=SVC(tol=5e-4)
    clf_3=AdaBoostClassifier(n_estimators=10)
    #clf_4=RandomForestClassifier(n_estimators=10,n_jobs=-1)
    
    #variants=[('l_svm',clf_1),
    #          ('svm',clf_2),
    #          ('ada',clf_3),
    #          ('rforest',clf_4)]
    variants=[('l_svm',clf_1),
              ('svm',clf_2),
              ('ada',clf_3)]
    
    ensemble_clf=VotingClassifier(estimators=variants,voting='hard').fit(X_train,y_train)
    t1_classifier=time.time()
    
    accuracy=round(100*ensemble_clf.score(X_test,y_test),4)
    train_time=round(t1_classifier-t0_classifier,4)
    
    save_model='ensemble_classifier.p'
    pickle.dump(ensemble_clf,open(save_model,'wb'))
    
    return ensemble_clf, accuracy, train_time 

def ensemble_nn(X_train,y_train,X_test,y_test):
    t0_classifier=time.time()
    
    clf_1=LinearSVC(penalty='l2')
    nn_model=KerasClassifier(build_fn=LeNet5,epochs=25,batch_size=100,verbose=0) 
    
    variants=[('ml_classifier_svm',clf_1),
              ('dl_classifier_lenet',nn_model)]
    
    ensemble_model=VotingClassifier(estimators=variants,voting='hard').fit(X_train,y_train)
    t1_classifier=time.time()
    
    accuracy=round(100*ensemble_model.score(X_test,y_test),4)
    train_time=round(t1_classifier-t0_classifier,4)
    
    save_model='dlml_ensemble_classifier.p'
    pickle.dump(ensemble_model,open(save_model,'wb'))
    return ensemble_model, accuracy, train_time