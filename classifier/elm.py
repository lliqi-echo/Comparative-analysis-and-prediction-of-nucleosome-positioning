import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LinearRegression
from elm import GenELMClassifier,GenELMRegressor
from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE 

data = np.loadtxt("work/DM/fcgr-2-mers.csv", dtype=np.float, delimiter=',')
np.random.shuffle(data)
x = data[:,0:16]
y = data[:,16]    

std_X = MinMaxScaler().fit_transform(x)

"""
X_std = StandardScaler().fit_transform(x)
transfer = PCA(n_components=0.85)
std_X = transfer.fit_transform(X_std)
print("PCA-size：", std_X.shape)
"""
final_test_acc = []
final_test_Sn = []
final_test_Sp =[]
final_mcc = []
final_auc = []
tprs = []
fprs = []

n_splits=20
sKF = StratifiedKFold(n_splits=n_splits, shuffle=False)
i = 0

stop_train = False
num_epochs = 10
for train_index, test_index in sKF.split(std_X,y):
    i +=1
    x_train = std_X[train_index]
    y_train = y[train_index]
    x_test = std_X[test_index]
    y_test = y[test_index]
    #-------------------------------------------------------------------------------
    grbf = GRBFRandomLayer(n_hidden=500, grbf_lambda=0.0001)
    act = MLPRandomLayer(n_hidden=500, activation_func='sigmoid') 
    rbf = RBFRandomLayer(n_hidden=290, rbf_width=0.0001, activation_func='sigmoid')
    
    clf = GenELMClassifier(hidden_layer=rbf) 
    clf.fit(x_train, y_train.ravel())
    y_pre = clf.predict(x_test)  
    y_score = clf.decision_function(x_test)
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    tprs.append(tpr)
    fprs.append(fpr)
    roc_auc = auc(fpr,tpr)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pre).ravel()
    test_acc = (tn + tp)/(tn + fp + fn + tp ) 
    test_Sn = tp/(fn+tp)
    test_Sp = tn/(fp+tn)
    mcc = (tp*tn-fp*fn)/pow(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),0.5) 
    final_test_acc.append(test_acc)
    final_test_Sn.append(test_Sn)
    final_test_Sp.append(test_Sp)
    final_mcc.append(mcc)
    final_auc.append(roc_auc)
    print('train_Accuracy: {:.3f}'.format(clf.score(x_train, y_train)))
    #print('test_Accuracy:%0.5f, test_Sn:%0.5f, test_Sp:%0.5f, mcc:%0.5f' % (test_acc, test_Sn, test_Sp, mcc))
    print('test_Accuracy:%0.5f, test_Sn:%0.5f, test_Sp:%0.5f, mcc:%0.5f, roc_auc:%0.5f' 
                                                   % (test_acc, test_Sn, test_Sp, mcc, roc_auc))
    print("confusion matrix:\n"+str (confusion_matrix(y_test, y_pre)))
    print('---------------------------------------------------')
               
Final_test_acc = (sum(final_test_acc) / len(final_test_acc)) 
Final_test_Sn = (sum(final_test_Sn) / len(final_test_Sn)) 
Final_test_Sp = (sum(final_test_Sp) / len(final_test_Sp)) 
Final_mcc =  (sum(final_mcc) / len(final_mcc)) 
Final_auc = (sum(final_auc)/len(final_auc)) 
print('Final_test_Accuracy:%0.5f' % (Final_test_acc))   
print('Final_test_Sn:%0.5f' % (Final_test_Sn)) 
print('Final_test_Sp:%0.5f' % (Final_test_Sp)) 
print('Final_mcc:%0.5f' % (Final_mcc)) 
print('Final_AUC:%0.5f' % (Final_auc))


