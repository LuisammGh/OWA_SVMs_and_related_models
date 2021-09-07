###########################################################################
#-- Two-Step SVM procedure with OWA in a ten fold cross validation
#-- Reference: 
#-- Redifining support vector machines with the ordered weighted average
#-- S. Maldonado, J. Merigó, J. Miranda
#-- Knowledge-Based Systems 148 (2018) 520 41--46.
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
import difflib

def ifnull(pos, val):
    l = len(sys.argv)
    if len(sys.argv) <= 1 or (sys.argv[pos] == None):
        return val
    return sys.argv[pos]


def peso(long,param,alpha):
    #0:Basic linguistic quantifier
    #1:Quadratic linguistic quantifier
    #2:Exponential linguistic quantifier NO ME CUADRA
    #3:Trigonometric linguistic quantifier
    weight=np.zeros(long)
    if(param==0):
        for i in range(0,long):
            weight[i]=pow((i+1)/long,alpha)-pow((i)/long,alpha)
    if(param==1):
        for i in range(0,long):
            weight[i]=1/(1-alpha*pow((i+1)/long,0.5))-1/(1-alpha*pow((i)/long,0.5))
    if(param==2):
        for i in range(0,long):
            weight[i]=np.exp(-alpha*(i+1)/long)-np.exp(-alpha*(i)/long)
    if(param==3):
         for i in range(0,long):
            weight[i]=np.arcsin(alpha*(i+1)/long)-np.arcsin(alpha*(i)/long)        
    #print(weight[i])
    media_pesos=np.mean(weight)
    weight=weight/media_pesos
    return(weight)
 

##----Parameters----------------------------------##
al=[0.2,0.4,0.6,0.8]
param=[0,1,2,3]
C = np.empty(15)
for i in range(0,15):
    C[i]=pow(2,7-i)
sigma = np.empty(15)
for i in range(0,15):
    sigma[i]=pow(2,7-i)
##-----------------------------------------------##

#-------------Data---------------------------------------------------------##
datafile = ifnull(1, "data/sonar_scale.txt")
features = int(ifnull(2,60))
details=ifnull(3,"details_sonar_MMM2018.txt")
summary=ifnull(4,"summary_sonar_MMM2018.txt")
data=sklearn.datasets.load_svmlight_file(datafile, n_features=features, 
                                    dtype=np.float64, multilabel=False, 
                                    zero_based='auto', query_id=False, offset=0, length=-1)
x=data[0]
y=data[1]
X=csr_matrix.toarray(x)
#---------------------------------------------------------------------------##

skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=2)
train=[]
test=[]
for train_index, test_index in skf.split(x, y):
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
f1.write("C alpha quantifier k objective sum_w b ACC AUC time1 time2\n")
f1.close()


feat=len(X[train[0]][0])
acc=np.zeros(10)
bal_acc=np.zeros(10)
av_time=np.zeros(10)
av_time0=np.zeros(10)
w=np.zeros(feat)

f2 = open(summary, "w")
f2.write("C alpha quantifier Acc AUC AvTime0 AvTime\n")
f2.close()


time0=time.time()
for ii in range(0,15):
    for jj in range(0,4):
        for alpha in range(0,4):
            for k in range(0,10):
                start_time0 = time.time()
                #---First step ------------------------------------------##
                SVMmodel= svm.SVC(kernel='linear', C=C[ii],random_state=None, probability=False)  
                SVMmodel.fit(X[train[k]],y[train[k]])
                av_time0[k] = time.time()-start_time0
                slack=np.zeros(len(train[k]))
                ind_slack=np.zeros(len(train[k]))
                predict=np.zeros(len(test[k]))
                slack2=np.zeros(len(train[k]))
                ind_slack2=np.zeros(len(train[k]))
                slack=1-y[train[k]]*SVMmodel.decision_function(X[train[k]])
                ind_slack=np.argsort(slack)[::-1]
                #-----------------------------------------------------------##
                n=len(train[k])
                start_time = time.time()
                #---Second step---------------------------------------------##
                opt_model = cpx.Model(name="SVM Model")
                w_vars  = {i: opt_model.continuous_var(lb=-opt_model.infinity,ub=opt_model.infinity,name="w_{0}".format(i)) for i in range(0,feat)}
                b_vars  = {i: opt_model.continuous_var(lb=-opt_model.infinity,ub=opt_model.infinity,name="b_{0}".format(i)) for i in range(0,1)}
                e_vars  = {i: opt_model.continuous_var(lb=0,ub=opt_model.infinity,name="e_{0}".format(i)) for i in range(0,n)}
                constraints1 = {i: opt_model.add_constraint(ct= y[train[k][i]]*(opt_model.sum(w_vars[j]*X[train[k][i]][j] for j in range(0,feat))
                                                                                +b_vars[0])+e_vars[i]>=1,ctname="constraints1_{0}".format(i)) for i in range(0,n)}
                objective = 1/2*opt_model.sum(w_vars[i]*w_vars[i] for i in range(0,feat))+(opt_model.sum(C[ii]*peso(len(train[k]),param[jj],al[alpha])[i]*e_vars[ind_slack[i]] for i in range(0,n)))    
                opt_model.minimize(objective) 
                opt_model.parameters.timelimit.set(1800)
                opt_model.solve()         
                status_model=opt_model.solve_details.status
                obj_val=objective.solution_value
                t1=time.time()
                #-----Classifier---------------------------------------------#
                for i in range(0,feat):
                   w[i]=w_vars[i].solution_value
                b=b_vars[0].solution_value

                for i in range(len(test[k])):
                    if(np.dot(w,X[test[k][i]].transpose())+b>0):
                        predict[i]=1
                    elif(np.dot(w,X[test[k][i]].transpose())+b<0):
                        predict[i]=-1
                t2=time.time()
                acc[k]=accuracy_score(y[test[k]],predict)
                bal_acc[k]=balanced_accuracy_score(y[test[k]],predict)
                av_time[k]=time.time()-start_time     
                f1=open(details,"a")
                f1.write("%.4f %.2f %d %d %.4f %.4f %.4f %.4f %.4f %f %f\n"
                 % (C[ii],al[alpha],param[jj],k,obj_val,sum(w),b,acc[k],bal_acc[k],av_time0[k],av_time[k]))
                f1.close()
            f2=open(summary,"a")
            f2.write("%.4f %.2f %d %.4f %.4f %.4f %.4f\n" %(C[ii],al[alpha],param[jj],np.mean(acc),np.mean(bal_acc),np.mean(av_time0),np.mean(av_time)))
            f2.close()




     