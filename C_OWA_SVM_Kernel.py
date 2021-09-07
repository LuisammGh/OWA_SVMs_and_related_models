###########################################################################
#-- C-OWA-SVM model with Gaussian Kernel in a ten fold cross validation
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


#---Parameters---------------------------------------------------------------#
al=[0.2,0.4,0.6,0.8]
param=[0,2] ## Choose weight by determining this parameter

def peso(long,param,alpha):
    #0:Basic linguistic quantifier
    #1:Quadratic linguistic quantifier
    #2:Exponential linguistic quantifier 
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
    weight2=weight[::-1]
    media_pesos=np.mean(weight)
    weight3=weight2/media_pesos
    return(weight3)

C = np.empty(15)

for i in range(0,15):
  C[i]=pow(2,7-i)

sigma = np.empty(15)
for i in range(0,15):
  sigma[i]=pow(2,7-i)

#---------------------------------------------------------------------------##

initime=time.time()

#---Data--------------------------------------------------------------------##
datafile = ifnull(1, "data/sonar_scale.txt")
features = int(ifnull(2,60))
details=ifnull(3,"details_sonar_C_OWA_SVM_GKernel.txt")
summary=ifnull(4,"summary_sonar_C_OWA_SVM_GKernel.txt")
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
f1.write("C sigma alpha quantifier k objective b ACC AUC time timesol \n")
f1.close()

X=csr_matrix.toarray(x)
feat=len(X[train[0]][0])
#-------------------------------------------------------------------------##

    
acc=np.zeros(10)
bal_acc=np.zeros(10)
av_time=np.zeros(10)
av_timesol=np.zeros(10)
  
t0=time.time()

f2 = open(summary, "w")
f2.write("C sigma alpha quantifier ACC AUC Average_time TotalTime time_sol\n")
f2.close() 

for ii in range(0,15): #C
    for ss in range(0,15): #sigma
        kernel=rbf_kernel(x,None,1/(2*pow(sigma[ss],2)))
        for jj in range(0,2):#weight
            for alpha in range(0,4): 
                for m in range(0,10):
                    bal_acc[m]=0
                    acc[m]=0
                for k in range(0,10): #Fold10               
                    n=len(train[k])
                    predict=np.zeros(len(test[k]))
                    start_time = time.time()##-----START TIME MODEL----------##
                    b=0
                    pesos=peso(len(train[k]),param[jj],al[alpha])
                    Cpesos=np.zeros(n)
                    Cpesos=C[ii]*pesos
                    opt_model = cpx.Model(name="SVM Model")
                    a_vars=opt_model.continuous_var_list(range(0,n),lb=0,name="a_")
                    eta_vars=opt_model.continuous_var_matrix(range(0,n),range(0,n),lb=0,ub=1,name="eta_vars_")         
                    cts4=[opt_model.sum(eta_vars[i,j] for j in range(0,n))==1 for i in range(0,n)]
                    cts5=[opt_model.sum(eta_vars[i,j] for i in range(0,n))==1 for j in range(0,n)]
                    cts3=[a_vars[i]-opt_model.sum(Cpesos[kk]*eta_vars[i,kk] for kk in range(0,n))<=0 for i in range(0,n)]
                    constraints=opt_model.add_constraint(ct=opt_model.sum(y[train[k][i]] * a_vars[i] for i in range(0,n)) == 0)
                    constraints4=opt_model.add_constraints(cts=cts4)
                    constraints5=opt_model.add_constraints(cts=cts5)
                    constraints3=opt_model.add_constraints(cts=cts3)                    
                    objective = opt_model.sum(a_vars[i] for i in range(0,n))-opt_model.sum(1/2*y[train[k][i]]*y[train[k][j]]*kernel[train[k][i]][train[k][j]]*a_vars[i]*a_vars[j]
                        for i in range(0,n) for j in range(0,n))                        
                    tsol1=time.time()
                    opt_model.maximize(objective)   
                    opt_model.parameters.timelimit.set(1800)
                    opt_model.solve()
                    tsol2=time.time()
                                 
                    status_model=opt_model.solve_details.status
                    obj_val=objective.solution_value
                
                    #----Clasifier----#
                    sol_a=np.zeros(n)
                    for i in range(0,n):
                        sol_a[i]=a_vars[i].solution_value
                    for i in range(0,n):
                        aux=0
                        if sol_a[i]>pow(10,-6):
                            for kk in range(0,n):
                                aux+= C[ii]*float(eta_vars[i,kk])*pesos[kk]
                            if float(a_vars[i])<aux:
                                aux1=0
                                for kk in range(0,n):
                                    if(float(a_vars[kk])>pow(10,-6)):
                                        aux1+=y[train[k][kk]]*float(a_vars[kk])*kernel[train[k][kk]][train[k][i]]
                                b=(1-y[train[k][i]]*aux1)/y[train[k][i]]
                                break
                
                    for i in range(len(test[k])):
                        aux1=0
                        for kk in range(0,n):
                            if(float(a_vars[kk])>pow(10,-6)):
                                aux1+=y[train[k][kk]]*float(a_vars[kk])*kernel[train[k][kk]][test[k][i]]
                        if(aux1+b>0):
                            predict[i]=1
                        elif(aux1+b<0):
                            predict[i]=-1
                    #--------------------accuracy and  AUC----------------#

                    acc[k]=accuracy_score(y[test[k]],predict)
                    bal_acc[k]=balanced_accuracy_score(y[test[k]],predict)
                    av_timesol[k]=tsol2-tsol1
                    av_time[k]=time.time()-start_time#----END TIME MODEL----#
                    
                    f1 = open(details, "a") 
                    f1.write("%.4f %.4f %.2f %d %d %.4f %.4f %.4f %.4f %f %f\n"
                             % (C[ii],sigma[ss],al[alpha],param[jj],k,obj_val,b,acc[k],bal_acc[k],av_time[k],av_timesol[k]))
                    f1.close()
                    
                f2 = open(summary, "a")   
                f2.write("%.4f %.4f %.2f %d %.4f %.4f %.4f %.4f %f\n"
                         % (C[ii],sigma[ss],al[alpha],param[jj],np.mean(acc),np.mean(bal_acc),np.mean(av_time),time.time()-initime,np.mean(av_timesol)))
                f2.close()

        

            
     