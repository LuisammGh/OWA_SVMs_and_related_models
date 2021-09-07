###########################################################################
#-- Classical l2-SVM model in a ten fold cross validation
###########################################################################

from __future__ import print_function
import sys
import numpy as np
import cplex
import sklearn
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from cplex.exceptions import CplexError
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
import docplex.mp.model as cpx
import docplex
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn import svm


def ifnull(pos, val):
    l = len(sys.argv)
    if len(sys.argv) <= 1 or (sys.argv[pos] == None):
        return val
    return sys.argv[pos]


datafile = ifnull(1, "data/sonar_scale.txt")
features = int(ifnull(2,60))
details=ifnull(3,"details_sonar_clasic.txt")
summary=ifnull(4,"summary_sonar_classic.txt")
data=sklearn.datasets.load_svmlight_file(datafile,
                                         n_features=features, 
                                    dtype=np.float64, multilabel=False, 
                                    zero_based='auto', query_id=False, offset=0, length=-1)

x=data[0]
y=data[1]
X=csr_matrix.toarray(x)
  
C = np.empty(15)
for i in range(0,15):
    C[i]=pow(2,7-i)



f2 = open(summary, "w")
f2.write("C Acc AUC Av_Fit_Time Av_Score_Time\n" )
f2.close()
    
scoring = ['accuracy', 'balanced_accuracy']  
cv10=StratifiedKFold(10,shuffle=True,random_state=2) 


train=[]
test=[]
for train_index, test_index in cv10.split(x, y):
    train.append(train_index)
    test.append(test_index)

f1 = open(details, "w")
for i in range(0,10):
    f1.write("%d " %(len(train[i])))
    for j in range(0,len(train[i])):
        f1.write("%d " %(train[i][j]))
    f1.write("\n")
f1.write("\n")
for i in range(0,10):
    f1.write("%d " %(len(test[i])))
    for j in range(0,len(test[i])):
        f1.write("%d " %(test[i][j]))
    f1.write("\n")
f1.write("\n")
f1.write("C k Acc AUC Fit_time Score_time\n") 
f1.close()

t0=time.time()
for ii in range(0,15):
    SVMmodel= svm.SVC(kernel='linear', C=C[ii],random_state=None, probability=False)   
    scores = cross_validate(SVMmodel, X, y, scoring=scoring,cv=cv10)
    f1=open(details,"a")
    for j in range(0,10):
        f1.write("%.4f %d %.4f %.4f %.4f %.4f\n" 
                 % (C[ii],j,scores['test_accuracy'][j],scores['test_balanced_accuracy'][j],scores['fit_time'][j],scores['score_time'][j]))
    f1.close()
    f2=open(summary,"a")
    f2.write("%.4f %.4f %.4f %.4f %.4f\n" 
             %(C[ii],np.mean(scores['test_accuracy']),np.mean(scores['test_balanced_accuracy']),np.mean(scores['fit_time']),np.mean(scores['score_time'])))
    f2.close()



     