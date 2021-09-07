###########################################################################
#-- NC-OWA-SVM model in a ten fold cross validation
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
import time
from sklearn import svm

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
    media_pesos=np.mean(weight)
    weight2=weight/media_pesos
    weight3=weight2[::-1]
    return(weight3)

C = np.empty(15)
for i in range(0,15):
    C[i]=pow(2,7-i)


#---------------------------------##

initime=time.time()

#---Datos-------------------------##
datafile = ifnull(1, "data/sonar_scale.txt")
features = int(ifnull(2,60))
details=ifnull(3,"details_sonar_NC_OWA.txt")
summary=ifnull(4,"summary_sonar_NC_OWA_SVM.txt")
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
f1.write("C alpha quantifier n k best_bound objective gap sum_w b ACC AUC time\n")               
f1.close()

X=csr_matrix.toarray(x)
feat=len(X[train[0]][0])
#---------------------------------##

    
acc=np.zeros(10)
bal_acc=np.zeros(10)
av_time=np.zeros(10)
av_timesol=np.zeros(10)

f2 = open(summary, "w")
f2.write("C alpha quantifier ACC AUC Average_time\n")          
f2.close()

t0=time.time()

for ii in range(0,15): #C 
    for jj in range(0,2):#param
        for alpha in range(0,4): #alpha
            for m in range(0,10):
                bal_acc[m]=0
                acc[m]=0
            for k in range(0,10): #Fold10    
                n=len(train[k])
                #---Initial solution --------------------------------#
                pesos=peso(n,param[jj],al[alpha])
                clf = svm.SVC(kernel='linear', C=C[ii],random_state=None, probability=False)
                clf.fit(X[train[k]], y[train[k]])             
                w_init=np.zeros(feat)
                for i in range(0,feat):
                    w_init[i]=clf.coef_[0][i]
                b_init=clf.intercept_[0]        
                desv=np.zeros(n)
                for i in range(0,n):
                    desv[i]=1-y[train[k][i]]*(np.dot(w_init,X[train[k][i]])+b_init)
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
                for i in range(0,feat):
                    obj_init+=1/2*w_init[i]*w_init[i]
                for i in range(0,n):
                    obj_init+=C[ii]*pesos[i]*theta_init[i]
                #----------------------------------------------------------#
                BigM=np.ceil(max(np.linalg.norm(X[train[k][i]]-X[train[k][j]]) for i in range(0,n) for j in range(0,n))*100)/100               
                predict=np.zeros(len(test[k]))
                start_time = time.time()##-----START TIME--MODEL-------##
                b=0
                Cpesos=np.zeros(n)
                Cpesos=C[ii]*pesos
                opt_model = cpx.Model(name="SVM Model")
                opt_model.parameters.timelimit.set(1800)
                bigM2=np.zeros(n)
                bigM2=C[ii]*pesos
                w_vars=opt_model.continuous_var_list(range(0,feat),lb=-opt_model.infinity,name="w")
                xi_vars=opt_model.continuous_var_list(range(0,n),lb=0,ub=BigM,name="xi")
                b_var=opt_model.continuous_var(lb=-opt_model.infinity,name="b")
                z_vars=opt_model.binary_var_matrix(range(0,n),range(0,n),name="z")
                theta_vars=opt_model.continuous_var_list(range(0,n),ub=BigM,name="theta")
                obj_var=opt_model.continuous_var_matrix(range(0,1),range(0,1),lb=0,name="obj")          
                objective = 1/2*opt_model.sum(w_vars[i]*w_vars[i] for i in range(0,feat))+C[ii]*opt_model.sum(pesos[kk]*theta_vars[kk] for kk in range(0,n))         
                cts1=[y[train[k][i]]*(opt_model.sum(w_vars[j]*X[train[k][i]][j] for j in range(0,feat))+b_var)>=1-xi_vars[i] for i in range(0,n)]
                constraints1=opt_model.add_constraints(cts=cts1,names="cts1_")
                cts2=[opt_model.sum(z_vars[i,kk] for kk in range(0,n))==1 for i in range(0,n)]
                constraints2=opt_model.add_constraints(cts=cts2,names="cts2_")
                cts3=[opt_model.sum(z_vars[i,kk] for i in range(0,n))==1 for kk in range(0,n)]
                constraints3=opt_model.add_constraints(cts=cts3,names="cts3_")
                cts4=[theta_vars[kk]>=xi_vars[i]-BigM*(1-opt_model.sum(z_vars[i,j] for j in range(0,kk+1))) for i in range(0,n) for kk in range(0,n)]           
                constraints4=opt_model.add_constraints(cts=cts4,names="cts4_")
                constraints5=opt_model.add_constraint(ct=objective<=obj_init)
                cts6=[theta_vars[kk]<=xi_vars[i]+BigM*(1-opt_model.sum(z_vars[i,j] for j in range(kk,n))) for i in range(0,n) for kk in range(0,n)]
                constraints6=opt_model.add_constraints(cts=cts6)       
                tsol1=time.time()
                opt_model.minimize(objective)            
                warmstart=opt_model.new_solution()
                for i in range(0,feat):
                    warmstart.add_var_value(w_vars[i],w_init[i])
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
                w=np.zeros(feat)
                for j in range(0,feat):
                  w[j]=float(w_vars[j])
                b=float(b_var)
                xi=np.zeros(n)
                for i in range(0,n):
                    xi[i]=float(xi_vars[i])
                for i in range(len(test[k])):
                    if(np.dot(w,X[test[k][i]].transpose())+b>0):
                        predict[i]=1
                    elif(np.dot(w,X[test[k][i]].transpose())+b<0):
                        predict[i]=-1
                acc[k]=accuracy_score(y[test[k]],predict)
                bal_acc[k]=balanced_accuracy_score(y[test[k]],predict)
                av_timesol[k]=tsol2-tsol1
                av_time[k]=time.time()-start_time#----END TIME--MODEL--------##

                f1 = open(details, "a")
                f1.write("%.4f %.2f %d %d %d %.4f %.4f %.4f %.4f %.4f %.4f %.4f %f\n"
                    % (C[ii],al[alpha],param[jj],n,k,opt_model.solve_details.best_bound,obj_val,opt_model.solve_details.mip_relative_gap,sum(w),b,acc[k],bal_acc[k],av_time[k]))
                f1.close()
            f2=open(summary,"a")
            f2.write("%.4f %.2f %d %.4f %.4f %.4f\n"
                % (C[ii],al[jj],param[jj],np.mean(acc),np.mean(bal_acc),np.mean(av_time)))
            f2.close()

f1.close()
f2.close()        

            
     
