import sklearn.ensemble as en
from sklearn.svm import SVC
from sklearn import tree
from sklearn import cross_validation as cxval
import sklearn as sk
import numpy as np

def readFile(fname, pred=False):
    '''Reads in a file for dataset.
    
    Args:
        fname: a string, the file name
        pred: the file we are reading has prediction. Default = False.
    
    Returns:
        xs: a list of list of feature values
        ys: a list of predictions
    '''

    xs = []
    ys = []
    with open(fname, 'r') as fin:
        # Consume the header line
        fin.readline()
        for line in fin:
            data = line.strip().split(',')
            
            if not pred:
                # The last value is always the prediction
                ys.append(float(data[-1]))

                xs.append([float(val) for val in data[:-1]])
            else:
                xs.append([float(val) for val in data])

    return xs, ys

def writeFile(fname, ys):
    '''Writes the prediction to file.
    
    Args:
        fname: the output file name
        ys: the list of predictions - a list of 0's and 1's
    '''
    with open(fname, 'w') as fout:
        fout.write('Id,Prediction\n')
        for i in range(len(ys)):
            fout.write("%d,%d\n" %(i+1, ys[i]))

def adaboost(n=50):
    num_cv = 10
    xs, ys = readFile('kaggle_train_wc.csv')
    ab = en.AdaBoostClassifier(n_estimators=n)
    scores = cxval.cross_val_score(ab, xs, ys, cv=num_cv)
    print "wc Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

    xs, ys = readFile('kaggle_train_tf_idf.csv')
    ab = en.AdaBoostClassifier(n_estimators=n)
    scores = cxval.cross_val_score(ab, xs, ys, cv=num_cv)
    print "tf_idf Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

def pred_test_ab(n=50):
    xs, ys = readFile('kaggle_train_wc.csv')
    ab = en.AdaBoostClassifier(n_estimators=n)
    ab.fit(xs, ys)

    xs, _ = readFile('kaggle_test_wc.csv', pred=True)
    ys = ab.predict(xs)
    writeFile('wc.csv', ys)

    xs, ys = readFile('kaggle_train_tf_idf.csv')
    ab = en.AdaBoostClassifier(n_estimators=n)
    ab.fit(xs, ys)

    xs, _ = readFile('kaggle_test_tf_idf.csv', pred=True)
    ys = ab.predict(xs)
    writeFile('tf.csv', ys)

def sv():
    num_cv = 10
    xs, ys = readFile('kaggle_train_wc.csv')
    clf = SVC()
    scores = cxval.cross_val_score(clf, xs, ys, cv=num_cv)
    print "wc Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

    xs, ys = readFile('kaggle_train_tf_idf.csv')
    clf= SVC()
    scores = cxval.cross_val_score(clf, xs, ys, cv=num_cv)
    print "tf_idf Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

def tr(**kwargs):
    num_cv = 10
    xs, ys = readFile('kaggle_train_wc.csv')
    clf = tree.DecisionTreeClassifier(**kwargs)
    scores = cxval.cross_val_score(clf, xs, ys, cv=num_cv)
    print "wc Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

    xs, ys = readFile('kaggle_train_tf_idf.csv')
    clf= tree.DecisionTreeClassifier(**kwargs)
    scores = cxval.cross_val_score(clf, xs, ys, cv=num_cv)
    print "tf_idf Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

def bagging(n=50):
    num_cv = 10
    xs, ys = readFile('kaggle_train_wc.csv')
    ab = en.BaggingClassifier(n_estimators=n)
    scores = cxval.cross_val_score(ab, xs, ys, cv=num_cv)
    print "wc Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

    xs, ys = readFile('kaggle_train_tf_idf.csv')
    ab = en.BaggingClassifier(n_estimators=n)
    scores = cxval.cross_val_score(ab, xs, ys, cv=num_cv)
    print "tf_idf Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

def forest(n=50):
    num_cv = 10
    xs, ys = readFile('kaggle_train_wc.csv')
    ab = en.RandomForestClassifier(n_estimators=n)
    scores = cxval.cross_val_score(ab, xs, ys, cv=num_cv)
    print "wc Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

    xs, ys = readFile('kaggle_train_tf_idf.csv')
    ab = en.RandomForestClassifier(n_estimators=n)
    scores = cxval.cross_val_score(ab, xs, ys, cv=num_cv)
    print "tf_idf Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

def grad(n=50):
    num_cv = 10
    xs, ys = readFile('kaggle_train_wc.csv')
    ab = en.GradientBoostingClassifier(n_estimators=n)
    scores = cxval.cross_val_score(ab, xs, ys, cv=num_cv)
    print "wc Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

    xs, ys = readFile('kaggle_train_tf_idf.csv')
    ab = en.GradientBoostingClassifier(n_estimators=n)
    scores = cxval.cross_val_score(ab, xs, ys, cv=num_cv)
    print "tf_idf Accuracy: %0.5f (+/- %0.5f)" %(scores.mean(), scores.std() * 2)

def pred_test_bag(n=50):
    xs, ys = readFile('kaggle_train_wc.csv')
    ab = en.BaggingClassifier(n_estimators=n)
    ab.fit(xs, ys)

    xs, _ = readFile('kaggle_test_wc.csv', pred=True)
    ys = ab.predict(xs)
    writeFile('wc_bag_%d.csv' %n, ys)

    xs, ys = readFile('kaggle_train_tf_idf.csv')
    ab = en.BaggingClassifier(n_estimators=n)
    ab.fit(xs, ys)

    xs, _ = readFile('kaggle_test_tf_idf.csv', pred=True)
    ys = ab.predict(xs)
    writeFile('tf_bag_%d.csv' %n, ys)







