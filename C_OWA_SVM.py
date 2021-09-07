###########################################################################
#-- C-OWA-SVM model in a ten fold cross validation
#-- Reference: 
#-- The soft-margin Support Vector Machine with ordered weighted average
#-- A.Marín, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía
#-- https://arxiv.org/abs/2107.06713
###########################################################################

from __future__ import print_function
import sys
import numpy as np
import cplex
import sklearn
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import time
from docplex.mp.model import Model

def ifnull(pos, val):
    l = len(sys.argv)
    if len(sys.argv) <= 1 or (sys.argv[pos] == None):
        return val
    return sys.argv[pos]

al=[0.2,0.4,0.6,0.8]
param=[2]

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
    media_pesos=np.mean(weight)
    weight2=weight/media_pesos
    weight3=weight2[::-1]
    return(weight3)


initime=time.time()
datafile = ifnull(1, "data/sonar_scale.txt")
features = int(ifnull(2,60))
details=ifnull(3,"details_sonar_C_OWA_SVM.txt")
summary=ifnull(4,"summary_sonar_C_OWA_SVM.txt")
data=sklearn.datasets.load_svmlight_file(datafile, n_features=features, 
                                    dtype=np.float64, multilabel=False, 
                                    zero_based='auto', query_id=False, offset=0, length=-1)

x=data[0]
y=data[1]
skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=2)
skf.get_n_splits(x, y)
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
f1.write("C alpha quantifier k objective sum_w b ACC AUC time\n")
f1.close()

    
C = np.empty(15)
for i in range(0,15):
  C[i]=pow(2,7-i)
    
acc=np.zeros(10)
bal_acc=np.zeros(10)
av_time=np.zeros(10)

al=[0.2,0.4,0.6,0.8]

X=csr_matrix.toarray(x)
feat=len(X[train[0]][0])


f2 = open(summary, "w")
f2.write("C alpha quantifier ACC AUC Average_time\n")
f2.close()


coef_obj=np.zeros((len(X),len(X)))
for i in range(0,len(X)):
    for j in range(0,len(X)):
        coef_obj[i][j]=y[i]*y[j]*np.dot(X[i],X[j].transpose())
        
t0=time.time()
for ii in range(0,15): #C 
    for jj in range(0,1):#param
        for alpha in range(0,4): 
            for m in range(0,10):
                bal_acc[m]=0
                acc[m]=0
            for k in range(0,10): #Fold
                n=len(train[k])
                solvar=np.zeros(n)
                solvar2=np.zeros((n,n))  
                predict=np.zeros(len(test[k]))
                start_time = time.time()##-----START TIME---------##
                b=0
                pesos=peso(len(train[k]),param[jj],al[alpha])
                Cpesos=np.zeros(n)
                Cpesos=C[ii]*pesos
                opt_model = cpx.Model(name="SVM Model")
                UB=C[ii]*sum(pesos)
                a_vars  = {i: opt_model.continuous_var(lb=0, ub=UB,name="a_{0}".format(i)) for i in range(0,n)}
                eta_vars = {(i,j):opt_model.continuous_var(ub=1) for i in range(n) for j in range(n)}
                constraints3={i : opt_model.add_constraint(ct=a_vars[i]-opt_model.sum(Cpesos[kk]*eta_vars[i,kk] for kk in range(0,n))<=0,
                    ctname="constraints3_{0}".format(i)) for i in range(0,n) }
                constraints = {j : opt_model.add_constraint(ct=opt_model.sum(y[train[k][i]] * a_vars[i] for i in range(0,n)) == 0,
                    ctname="constraint_{0}".format(j)) for j in range(0,1)}        
                constraints4={i : opt_model.add_constraint(ct=opt_model.sum(eta_vars[i,j] for j in range(0,n))==1,
                    ctname="constraints4_{0}".format(i)) for i in range(0,n)}
                constraints5={j : opt_model.add_constraint(ct=opt_model.sum(eta_vars[i,j] for i in range(0,n))==1,
                    ctname="constraints5_{0}".format(j)) for j in range(0,n)} 
                objective = opt_model.sum(a_vars[i] for i in range(0,n))-opt_model.sum(1/2*coef_obj[train[k][i]][train[k][j]]*a_vars[i]*a_vars[j]
                    for i in range(0,n) for j in range(0,n))            
                opt_model.maximize(objective)            
                opt_model.parameters.timelimit.set(1800)
                opt_model.solve()        
                status_model=opt_model.solve_details.status
                obj_val=objective.solution_value
                for i in range(0,n):
                    solvar[i]=a_vars[i].solution_value
                for i in range(0,n):
                    for j in range(0,n):
                        solvar2[i][j]=eta_vars[i,j].solution_value
                w=np.zeros(feat)
                for i in range(0,n):
                    for j in range(0,feat):
                        w[j]+=solvar[i]*y[train[k][i]]*X[train[k][i]][j]
                for i in range(0,n):
                    aux=0
                    if solvar[i]>pow(10,-6):
                        for kk in range(0,n):
                            aux+= C[ii]*solvar2[i][kk]*pesos[kk]
                        if float(solvar[i])<aux:
                            b=1/y[train[k][i]]-np.dot(X[train[k][i]],w.transpose())
                            break 
                #--------------------accuracy AUC----------------#
                for i in range(len(test[k])):
                    if(np.dot(w,X[test[k][i]].transpose())+b>0):
                        predict[i]=1
                    elif(np.dot(w,X[test[k][i]].transpose())+b<0):
                        predict[i]=-1

                acc[k]=accuracy_score(y[test[k]],predict)
                bal_acc[k]=balanced_accuracy_score(y[test[k]],predict)                
                acc_prueba=(np.sum(acc)+10-k-1)/10
                balacc_prueba=(np.sum(bal_acc)+10-k-1)/10          
                av_time[k]=time.time()-start_time#----END TIME------##

                f1 = open(details, "a")
                f1.write("%.4f %.2f %d %d %.4f %.4f %.4f %.4f %.4f %f\n"
                    % (C[ii],al[alpha],param[jj],k,obj_val,sum(w),b,acc[k],bal_acc[k],av_time[k]))
                f1.close()
            f2=open(summary,"a")
            f2.write("%.4f %.2f %d %.4f %.4f %.4f\n"
                % (C[ii],al[jj],param[jj],np.mean(acc),np.mean(bal_acc),np.mean(av_time)))
            f2.close()
     

            
     