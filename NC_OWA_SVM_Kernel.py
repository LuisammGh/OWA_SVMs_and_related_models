###########################################################################
#-- NC-OWA-SVM model with Gaussian Kernel in a ten fold cross validation
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
#import matplotlib.pyplot as plt
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
from sklearn.metrics import roc_auc_score
import time
from sklearn import svm
import os

from cplex.callbacks import LazyConstraintCallback
from docplex.mp.callbacks.cb_mixin import *
from docplex.mp.model import Model
from sklearn.model_selection import train_test_split


def ifnull(pos, val):
    l = len(sys.argv)
    if len(sys.argv) <= 1 or (sys.argv[pos] == None):
        return val
    return sys.argv[pos]

#---Parameters----------------#
al=[0.2,0.4,0.6,0.8]
param=[1,3]

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
    if(param==4):
         for i in range(0,round(long/3)):
             weight[i]=0.1
         for i in range(round(long/3),round(2*long/3)):
             weight[i]=1
         for i in range(round(2*long/3),long):
             weight[i]=0.1
    media_pesos=np.mean(weight)
    weight2=weight/media_pesos
    weight3=weight2[::-1]
    return(weight3)

C = np.empty(15)
for i in range(0,15):
    C[i]=pow(2,7-i)

sigma = np.empty(15)
for i in range(0,15):
    sigma[i]=pow(2,7-i)
        
#---------------------------------##

initime=time.time()

#---Data-------------------------##
datafile = ifnull(1, "data/sonar_scale.txt")
features = int(ifnull(2,60))
details=ifnull(3,"details_sonar_NC_OWA_SVM_GKernel.txt")
summary=ifnull(4,"summary_sonar_NC_OWA_SVM_GKernel.txt")
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
f1.close()

X=csr_matrix.toarray(x)
feat=len(X[train[0]][0])

acc=np.zeros(10)
bal_acc=np.zeros(10)
av_time=np.zeros(10)
av_timesol=np.zeros(10)

t0=time.time()

f1=open(details,"a")
f1.write("C sigma alpha quantifier n k best_bound objective gap b ACC bal_acc time time_b time_Big \n")
f1.close()

f2=open(summary,"w")
f2.write("C sigma alpha quantifier ACC bal_acc Average_time \n")
f2.close()

for ii in range(0,15): #C 
    for s in range(0,15):
        kernel=rbf_kernel(x,None,1/(2*pow(sigma[s],2)))
        for jj in range(0,2):#param3
            for alpha in range(0,4): #alpha4
                for m in range(0,5):
                    bal_acc[m]=0
                    acc[m]=0
                for k in range(0,10): #Fold10    
                    n=len(train[k])
                    pesos=peso(n,param[jj],al[alpha])
                      
                    clf = svm.SVC(kernel='rbf',gamma=1/(2*pow(sigma[s],2)),C=C[ii],random_state=None, probability=False)
                    clf.fit(X[train[k]][0:n], y[train[k]][0:n])
                    
                    #--- Building an initial feasible solution ---------------#
                    
                    a_init=np.zeros(n)
                    for j in range(0,len(clf.dual_coef_[0])):
                        a_init[clf.support_[j]]=abs(clf.dual_coef_[0][j])
                    b_init=clf.intercept_[0]
                
                    desv=np.zeros(n)
                    for i in range(0,n):
                        aux=0
                        for j in range(0,n):
                            aux+=y[train[k][j]]*a_init[j]*kernel[train[k][j]][train[k][i]]
                        aux+=b_init
                        desv[i]=1-y[train[k][i]]*(aux)
                    orden=np.argsort(desv)

                    z_init=np.zeros((n,n))
                    for i in range(0,n):
                        z_init[orden[i]][i]=1
                
                    xi_init=np.zeros(n)
                    for i in range(0,n):
                        if(desv[i]>0):
                            xi_init[i]=desv[i]
                        else:
                            xi_init[i]=0
                    theta_init=np.zeros(n)
                    for i in range(0,n):
                        theta_init[i]=xi_init[orden[i]]

                    obj_init=0
                    for i in range(0,n):
                        for j in range(0,n):
                            obj_init+=1/2*y[train[k][i]]*y[train[k][j]]*a_init[i]*a_init[j]*kernel[train[k][i]][train[k][j]]
                    for i in range(0,n):
                        obj_init+=C[ii]*pesos[i]*theta_init[i]

                    #----------Bounds on b--------------------------#
                    tt1=time.time()
                    with Model() as opt_model2:
                        b_var=opt_model2.continuous_var(lb=-opt_model2.infinity,name="b")
                        a_vars=opt_model2.continuous_var_list(range(0,n),lb=0,ub=C[ii]*np.sum(pesos),name="a")
                        xi_vars=opt_model2.continuous_var_list(range(0,n),lb=0,name="xi")
                        theta_var=opt_model2.continuous_var(lb=0,name="theta")   
                        cts1=[y[train[k][i]]*(opt_model2.sum(y[train[k][kk]]*kernel[train[k][kk]][train[k][i]]*a_vars[kk] for kk in range(0,n))+b_var)>=1-xi_vars[i] for i in range(0,n)]
                        constraints1=opt_model2.add_constraints(cts=cts1)
                        constraints2=opt_model2.add_constraint(ct=opt_model2.sum(y[train[k][i]]*a_vars[i] for i in range(0,n))==0)
                        cts3=[xi_vars[i]<=theta_var for i in range(0,n)]
                        constraints3=opt_model2.add_constraints(cts=cts3)
                        constraint4=opt_model2.add_constraint(1/2*opt_model2.sum(y[train[k][i]]*y[train[k][j]]*kernel[train[k][i]][train[k][j]]*a_vars[i]*a_vars[j] for i in range(0,n) for j in range(0,n))+C[ii]*pesos[n-1]*theta_var<=obj_init)                                                         
                        opt_model2.maximize(b_var)
                        opt_model2.solve()
                        status_model2=opt_model2.solve_details.status
                        b_max=b_var.solution_value                       
                        opt_model2.parameters.threads=1
                        opt_model2.minimize(b_var)
                        opt_model2.solve()
                        b_min=b_var.solution_value
                    
                    tt2=time.time()
                     #----------Feasible M parameter------------------------#
                    tt3=time.time()
                    with Model() as opt_model2:
                        b_var=opt_model2.continuous_var(lb=b_min,ub=b_max,name="b")
                        a_vars=opt_model2.continuous_var_list(range(0,n),lb=0,ub=C[ii]*np.sum(pesos),name="a")
                        xi_vars=opt_model2.continuous_var_list(range(0,n),lb=0,name="xi")
                        theta_var=opt_model2.continuous_var(lb=0,name="theta")
                        cts1=[y[train[k][i]]*(opt_model2.sum(y[train[k][kk]]*kernel[train[k][kk]][train[k][i]]*a_vars[kk] for kk in range(0,n))+b_var)>=1-xi_vars[i] for i in range(0,n)]
                        constraints1=opt_model2.add_constraints(cts=cts1)
                        constraints2=opt_model2.add_constraint(ct=opt_model2.sum(y[train[k][i]]*a_vars[i] for i in range(0,n))==0)
                        cts3=[xi_vars[i]<=theta_var for i in range(0,n)]
                        constraints3=opt_model2.add_constraints(cts=cts3)
                        constraint4=opt_model2.add_constraint(1/2*opt_model2.sum(y[train[k][i]]*y[train[k][j]]*kernel[train[k][i]][train[k][j]]*a_vars[i]*a_vars[j] for i in range(0,n) for j in range(0,n))+C[ii]*pesos[n-1]*theta_var<=obj_init)                
                        opt_model2.maximize(theta_var)
                        opt_model2.solve()
                        Big=theta_var.solution_value
                        tt4=time.time()
                   
                    #--- Solving the model --------------------------------  #
                                               
                    predict=np.zeros(len(test[k]))
                    start_time = time.time()##-----START TIME--Model-------##
                    b=0
                    Cpesos=np.zeros(n)
                    Cpesos=C[ii]*pesos
                    opt_model = cpx.Model(name="SVM Model")                               
                    a_vars=opt_model.continuous_var_list(range(0,n),ub=C[ii]*np.sum(pesos),name="a")
                    xi_vars=opt_model.continuous_var_list(range(0,n),lb=0,ub=Big,name="xi")
                    b_var=opt_model.continuous_var(lb=b_min,ub=b_max,name="b")
                    z_vars=opt_model.binary_var_matrix(range(0,n),range(0,n),name="z")
                    theta_vars=opt_model.continuous_var_list(range(0,n),ub=Big,name="theta")
                     
                    objective = 1/2*opt_model.sum(y[train[k][i]]*y[train[k][j]]*kernel[train[k][i]][train[k][j]]*a_vars[i]*a_vars[j] for i in range(0,n) for j in range(0,n))+C[ii]*opt_model.sum(pesos[kk]*theta_vars[kk] for kk in range(0,n))                
                
                    cts1=[y[train[k][i]]*(opt_model.sum(y[train[k][kk]]*kernel[train[k][kk]][train[k][i]]*a_vars[kk] for kk in range(0,n))+b_var)>=1-xi_vars[i] for i in range(0,n)]
                    constraints1=opt_model.add_constraints(cts=cts1,names="cts1_")
                    cts3=[opt_model.sum(z_vars[i,kk] for i in range(0,n))==1 for kk in range(0,n)]
                    constraints3=opt_model.add_constraints(cts=cts3,names="cts3_")
                    cts4=[theta_vars[kk]>=xi_vars[i]-Big*(1-opt_model.sum(z_vars[i,j] for j in range(0,kk+1))) for i in range(0,n) for kk in range(0,n)]
                    constraints4=opt_model.add_constraints(cts=cts4,names="cts4_")
                    constraints5=opt_model.add_constraint(ct=objective<=obj_init+0.001)       
                    tsol1=time.time()
                    opt_model.parameters.timelimit.set(1800)
                    opt_model.minimize(objective)    


                    warmstart=opt_model.new_solution()
                    for i in range(0,n):
                        warmstart.add_var_value(a_vars[i],a_init[i])
                    warmstart.add_var_value(b_var,b_init)
                    for i in range(0,n):
                        for j in range(0,n):
                            warmstart.add_var_value(z_vars[i,j],z_init[i][j])
                    for kk in range(0,n):
                        warmstart.add_var_value(theta_vars[kk],theta_init[kk])
                    for i in range(0,n):
                        warmstart.add_var_value(xi_vars[i],xi_init[i])
                    opt_model.add_mip_start(warmstart)
                                
                    opt_model.solve()
                    tsol2=time.time()                 
                
                    status_model=opt_model.solve_details.status
                    obj_val=objective.solution_value
                    b=b_var.solution_value
                    
                    for i in range(len(test[k])):
                        aux1=0
                        for kk in range(0,n):
                            if(float(a_vars[kk])>pow(10,-6)):
                                aux1+=y[train[k][kk]]*float(a_vars[kk])*kernel[train[k][kk]][test[k][i]]
                        if(aux1+b>0):
                            predict[i]=1
                        elif(aux1+b<0):
                            predict[i]=-1
                    
                    acc[k]=accuracy_score(y[test[k]],predict)
                    bal_acc[k]=balanced_accuracy_score(y[test[k]],predict)
                    av_timesol[k]=tsol2-tsol1
                    av_time[k]=time.time()-start_time#----END TIME--Model---##
                               
                    f1 = open(details, "a")
                    f1.write("%.4f %.4f %.2f %d %d %d %.4f %.4f %.4f %.4f %.4f %.4f %f %.4f %.4f\n"
                             % (C[ii],sigma[s],al[alpha],param[jj],n,k,opt_model.solve_details.best_bound,obj_val,opt_model.solve_details.mip_relative_gap,b,acc[k],bal_acc[k],av_time[k],tt2-tt1,tt4-tt3))
                                     
                    f1.close()
                     
                
                f2 = open(summary, "a")
                f2.write("%.4f %.4f %.2f %d %.4f %.4f %.4f\n"
                          % (C[ii],sigma[s],al[alpha],param[jj],np.mean(acc),np.mean(bal_acc),np.mean(av_time)))
                f2.close()

       

            
     
