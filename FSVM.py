import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix

from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

Banknote = pd.read_csv ('C://Users//Parisan.Sh//Desktop//pattern//banknote.dataset.csv', encoding='ansi')
#print(Banknote)

y = Banknote.y
#print(y)

Banknote = Banknote.drop('y', axis = 1)
Banknote_Normalize = scale(Banknote)

Banknote = pd.DataFrame(Banknote_Normalize , index = Banknote.index , columns = Banknote.columns )
#print(Banknote)

x = Banknote_Normalize
x_train , x_test , y_train , y_test = train_test_split (x , y ,test_size = .2 , random_state = 42)

#_______________ SVM ________________

svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train,y_train)

y_predict = svclassifier.predict(x_test)

#Evaluating the SVM
print('\nThe Confusion Matrix:')
print(confusion_matrix(y_test,y_predict))  
print(classification_report(y_test,y_predict))

w = svclassifier.coef_
bias = svclassifier.intercept_

print('w = ', w)
print('b = ',bias)
"""print('Indices of support vectors = \n', svclassifier.support_)#for more result u can uncmnt it 
print('Support vectors = \n', svclassifier.support_vectors_)
print('Number of support vectors for each class = \n', svclassifier.n_support_)
print('Coefficients of the support vector in the decision function = \n', np.abs(svclassifier.dual_coef_))""" 
#______________ Hard SVM _____________

#Implementation using CVXOPT
#Initializing values and computing H

y_train = y_train.values
y_train = y_train.reshape(-1,1) * 1.

m , n = x_train.shape
x_dash = y_train * x_train
H = np.dot(x_dash , x_dash.T) * 1.

#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m,1)))
G = cvxopt_matrix(-np.eye(m))  #np.eye(m) -> Return a m-D array with ones on the diagonal and zeros elsewhere.
h =cvxopt_matrix(np.zeros(m))
A = cvxopt_matrix(y_train.reshape(1,-1))
b =cvxopt_matrix(np.zeros(1))

#Setting solver parameters 
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

#w parameter in vectorized form
w = ((y_train * alphas).T @ x_train).reshape(-1,1)

#Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

#Computing bias
bias = y_train[S] - np.dot(x_train[S], w)
r , c = bias.shape
b_avg = sum(bias) / r 
#Display results
#print('Alphas = ',alphas[alphas > 1e-4]) #for more result u can uncmnt it 
print('\nHard margin result:\nw = ', w.flatten())
print('b = ', b_avg)

#____________ Soft SVM _____________

# m , n , y_trin , x_dash , H are the same as hard margine
C = 5

#Converting into cvxopt format - as previously
# p , q , A , b are the same as hard margine
G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m)))) #np.vstack()-> using two or more arrays that allows you to combine arrays and make them into one array. Vstack stands for vertical stack
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C))) #Hstack stands for horizontal stack

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

#Computing and printing parameters
w = ((y_train * alphas).T @ x_train).reshape(-1,1)
S = (alphas > 1e-4).flatten()
bias = y_train[S] - np.dot(x_train[S], w)
#print('Alphas = ',alphas[alphas > 1e-4]) #for more result u can uncmnt it 
print('\nsoft margin result:\nw = ', w.flatten())
print('b = ', bias[0])

r , c = bias.shape
b_avg = sum(bias) / r

w = w.T
y_res = np.dot(w , x_test.T)

y_res = y_res.T
b_  = np.ones((275,1))*b_avg
y_res += b_
y_res = np.sign(y_res)

print('\nThe Confusion Matrix:')
print(confusion_matrix(y_test,y_res))
print(classification_report(y_test,y_res))

#____________ Fuzzy SVM _____________
#Use Class Center to Reduce the Effects of Outliers

#at first we should find mean of class 1 and -1
mean_1 = np.zeros((1,4))
mean_0 = np.zeros((1,4))
count_1 = 0
count_0 = 0
sigma = 0.02

for sample in range(m):
    if y[sample]== 1 :
        count_1 += 1
        for feature in range(n):
            mean_1[0][feature] += x_train[sample][feature]
    else:
        count_0 +=1
        for feature in range(n):
            mean_0[0][feature] += x_train[sample][feature]

mean_1 /= count_1
mean_0 /= count_0

#Find radius of class 1 and -1
r_1 = 0
r_0 = 0
for sample in range(m):
    current_r = 0
    if y[sample]== 1 :
        for feature in range(n):
            current_r += np.math.pow(x_train[sample][feature],2)
        current_r = np.math.sqrt(current_r)    
        if current_r > r_1 :
            r_1 = current_r
    else:
        for feature in range(n):
            current_r += np.math.pow(x_train[sample][feature],2)
        current_r = np.math.sqrt(current_r)
        if current_r > r_0 :
            r_0 = current_r

# compute fuzzy membership for each train point
fuzzy_membership = np.zeros(m)
for sample in range(m):
    temp = 0
    if y[sample]== 1 :
        for feature in range(n):
            current_feature = 0
            current_feature = mean_1[0][feature] - x_train[sample][feature]
            current_feature = np.math.pow(current_feature ,2)
            temp += current_feature
        temp = np.math.sqrt(temp)
        temp /= (r_1 + sigma)
        fuzzy_membership[sample] = 1 - temp
    else:
        for feature in range(n):
            current_feature = 0
            current_feature = mean_0[0][feature] - x_train[sample][feature]
            current_feature = np.math.pow(current_feature ,2)
            temp += current_feature
        temp = np.math.sqrt(temp)
        temp /= (r_0 + sigma)
        fuzzy_membership[sample] = 1 - temp

# m , n , y_trin , x_dash , H are the same as SVM
C = 5

#Converting into cvxopt format - as previously
# p , q , A , b are the same as SVM
G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m)))) 
h = cvxopt_matrix(np.hstack((np.zeros(m), fuzzy_membership * C))) #used fuzzy membership for each point

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

#Computing and printing parameters
w = ((y_train * alphas).T @ x_train).reshape(-1,1)
S = (alphas > 1e-4).flatten()
bias = y_train[S] - np.dot(x_train[S], w)
#print('Alphas = ',alphas[alphas > 1e-4]) #for more result u can uncmnt it 
print('\nFuzzy SVM result:\nw = ', w.flatten())
print('b = ', bias[0])

r , c = bias.shape
b_avg = sum(bias) / r

w = w.T
y_res = np.dot(w , x_test.T)

y_res = y_res.T
b_  = np.ones((275,1))*b_avg
y_res += b_
y_res = np.sign(y_res)

print('\nThe Confusion Matrix:')
print(confusion_matrix(y_test,y_res))
print(classification_report(y_test,y_res))