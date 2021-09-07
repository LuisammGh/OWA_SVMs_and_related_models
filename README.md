# OWA_SVMs_and_related_models

This repository contains the following codes:
1.  C_OWA_SVM.py: C-OWA-SVM model in a ten fold cross validation. The description of this model can be found in: 
    The soft-margin Support Vector Machine with ordered weighted average
    A.Marín, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía
    https://arxiv.org/abs/2107.06713
2.  C_OWA_SVM_Kernel.py: C-OWA-SVM model using the Gaussian Kernel in a ten fold cross validation. The description of this model can be found in: 
    The soft-margin Support Vector Machine with ordered weighted average
    A.Marín, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía
    https://arxiv.org/abs/2107.06713
3.  NC_OWA_SVM.py: NC-OWA-SVM model in a ten fold cross validation. The description of this model can be found in: 
    The soft-margin Support Vector Machine with ordered weighted average
    A.Marín, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía
    https://arxiv.org/abs/2107.06713
4.  CN_OWA_SVM_Kernel.py: NC-OWA-SVM model using the Gaussian Kernel in a ten fold cross validation. The description of this model can be found in: 
    The soft-margin Support Vector Machine with ordered weighted average
    A.Marín, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía
    https://arxiv.org/abs/2107.06713
5.  Classical_LIBSVM.py: Classical l2-SVM model in a ten fold cross validation. See:
    Feature selection via concave minimization and support vector machines,P. S. Bradley, O. L. Mangasarian, ICML 98 (1998) 82-90.
6.  Classical_LIBSVM_Kernel.py: Classical l2-SVM model using the Gaussian Kernel in a ten fold cross validation. See:
    Feature selection via concave minimization and support vector machines,P. S. Bradley, O. L. Mangasarian, ICML 98 (1998) 82-90.
7.  MMM2018.py: Two-Step SVM procedure with OWA in a ten fold cross validation. The description of this procedure can be found in:
    Redifining support vector machines with the ordered weighted average
    S. Maldonado, J. Merigó, J. Miranda
    Knowledge-Based Systems 148 (2018) 520 41--46.
8.  MMM2018_Kernel.py: Two-Step SVM procedure with OWA using Gaussian Kernel in a ten fold cross validation. The description of this procedure can be found in:
    Redifining support vector machines with the ordered weighted average
    S. Maldonado, J. Merigó, J. Miranda
    Knowledge-Based Systems 148 (2018) 520 41--46.
    
These codes are implemented in Python and solved using Cplex. The user provides the following data:
-   Name of the data file. Example: data/sonar_scale.txt 
-   Number of features of the dataset. Example: 60
-   Name of the text file where the details about each fold will be written. Example: details_sonar.txt
-   Name of the text file where a summary of the results will be written. Example: summary_sonar.txt.

The used parameter values are the following:
-   C=2^{-7},2^{-6},...,2^{7}
-   sigma=2^{-7},2^{-6},...,2^{7}
-   OWA weights based on linguistic quantifiers: basic, quadratic, exponential and trigonometric
-   alpha= 0.2,0.4,0.6,0.8

Besides, folder "data" include three examples of datasets, particularly: australian_scale, ionosphere_scale, sonar_scale, and splice_scale. More datasets can be found in:
Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011. 
Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm.



