from __future__ import print_function
import paddle
from PIL import Image
import paddle.fluid as fluid
import numpy as np
import sys
import os 
import shutil
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def custom_image_reader(file_list, data_dir):
    q, p=file_list.shape
    image_count = q
    label_set = set()
    for i in range(q):
        img_path, label = file_list[i]
        img_path = os.path.join(data_dir, img_path)
        label_set.add(label)
        class_dim = len(label_set)
    print("class dim:{0} image count:{1}".format(class_dim, image_count))
    
    def reader():
        for i in range(q):
            img_path, label = file_list[i]
            img_path = os.path.join(data_dir, img_path)
            img = Image.open(img_path).convert('L')
            img = img.resize((64, 64), Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = abs(img/255-1)            
            yield img, int(label)

    return reader


def CNN(image):
    conv1_1 = fluid.layers.conv2d(input=image, num_filters=64, filter_size=3, act='relu', stride=1) 
    conv1 = fluid.layers.batch_norm(input=conv1_1, act='relu')
    print(conv1.shape)
    pool1 = fluid.layers.pool2d(input=conv1, pool_size=2,pool_type='max',pool_stride=2)
    print(pool1.shape)
 
    conv2_1 = fluid.layers.conv2d(input=pool1, num_filters=32, filter_size=3, act='relu', stride=1) 
    conv2 = fluid.layers.batch_norm(input=conv2_1, act='relu')
    print(conv2.shape)
    pool2 = fluid.layers.pool2d(input=conv2, pool_size=2,pool_type='max',pool_stride=2)         
    print(pool2.shape)         
                         
    h1 = fluid.layers.fc(input=pool2, size=100, act='relu')  
    #h1 =  fluid.layers.batch_norm(input=h1, act='relu') 
    h1 = fluid.layers.dropout(x=h1, dropout_prob=0.5, seed=None)
    h2 = fluid.layers.fc(input=h1, size=50, act='relu')
    #h2 =  fluid.layers.batch_norm(input=h2, act='relu') 
    h2 = fluid.layers.dropout(x=h2, dropout_prob=0.5, seed=None)     
    prediction = fluid.layers.fc(input=h2, size=2, act='softmax') 
    return prediction



final_test_acc = []
final_test_Sn = []
final_test_Sp =[]
final_mcc = []
final_auc = []
tprs = []
fprs = []
data_dir1 = "work/FCGR-4-mers"

humandata = np.loadtxt("work/FCGR-4-mers--list.txt", dtype=np.str, delimiter=' ')
np.random.shuffle(humandata)
n_splits=10
sKF = StratifiedKFold(n_splits=n_splits, shuffle=False)
i = 0

stop_train = False
num_epochs = 20
doc = open('work/print.txt','w')
for train_index, test_index in sKF.split(humandata[:,0],humandata[:,1]):
    i +=1
    train_program = fluid.Program()
    startup_prog = fluid.Program() 
    with fluid.program_guard(train_program, startup_prog):
        image = fluid.layers.data(name='image', shape=[1, 64, 64], dtype='float64')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')        
        with fluid.unique_name.guard():
            predict = CNN(image=image)    
            cost = fluid.layers.cross_entropy(input=predict, label=label) 
            avg_cost = fluid.layers.mean(cost)
            acc = fluid.layers.accuracy(input=predict, label=label)           
            test_program = train_program.clone(for_test=True)       
            optimizer = fluid.optimizer.AdamaxOptimizer(learning_rate=0.001,
                                                       regularization=fluid.regularizer.L2Decay(regularization_coeff=0.005)) 
            #optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.001, momentum=0.98)                                                                   
            opts = optimizer.minimize(avg_cost)


    use_gpu=False
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    #place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 进行参数初始化
    exe.run(startup_prog)    
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    model_save_dir = str("/home/aistudio/model/dna%s.inference.model" %i)
    train_d = humandata[train_index]
    test_dna = humandata[test_index]
    train_dna, val_dna =train_test_split(train_d,test_size=0.1)               
    train_reader = paddle.batch(custom_image_reader(train_dna, data_dir1), batch_size=64)
    val_reader = paddle.batch(custom_image_reader(val_dna, data_dir1), batch_size=64)  
    test_reader = paddle.batch(custom_image_reader(test_dna, data_dir1), batch_size=64)
    test_label = list(test_dna[:,1])
    test_label = list(map(int, test_label))
    predict_label = []
    score_test = []
    stop_train = False
    Valacc=[]
    Valloss=[]
    Trainacc=[]
    Trainloss=[]
    #successive_count = 0
    for pass_id in range(num_epochs):
        train_reader=paddle.reader.shuffle(train_reader, 3000)
        for step_id, data in enumerate(train_reader()):
            train_acc, loss = exe.run(program=train_program,
                                        feed=feeder.feed(data),                    
                                        fetch_list=[acc,avg_cost])          
            Trainacc.append(train_acc[0])
            Trainloss.append(loss[0])                
        Tacc=(sum(Trainacc) / len(Trainacc)) 
        Tloss=(sum(Trainloss) / len(Trainloss))   
        print("Pass:{0}, train_acc:{1}, train_loss:{2}" .format(pass_id, Tacc, Tloss)) 
        
        
    fluid.io.save_inference_model(dirname=model_save_dir,
                                  feeded_var_names=['image'],
                                  target_vars=[predict], 
                                  executor=exe,
                                  main_program=train_program) 

    for step, data in enumerate(test_reader()):                         
        out = exe.run(program=test_program, 
                            feed=feeder.feed(data),                    
                            fetch_list=[predict])           
        for item in out:
            for it in item:
                score_test.append(it[1])
                predict_label.append(np.argmax(it))
    fpr, tpr, thresholds = roc_curve(test_label,score_test)
    tprs.append(tpr)
    fprs.append(fpr)
    roc_auc = auc(fpr,tpr)            
    tn, fp, fn, tp = confusion_matrix(test_label, predict_label).ravel()
    test_acc = (tn + tp)/(tn + fp + fn + tp ) 
    test_Sn = tp/(fn+tp)
    test_Sp = tn/(fp+tn)
    mcc = (tp*tn-fp*fn)/pow(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),0.5) 
    final_test_acc.append(test_acc)
    final_test_Sn.append(test_Sn)
    final_test_Sp.append(test_Sp)
    final_mcc.append(mcc)
    final_auc.append(roc_auc)
    print('test_Accuracy:%0.5f, test_Sn:%0.5f, test_Sp:%0.5f, mcc:%0.5f, roc_auc:%0.5f' 
                                                   % (test_acc, test_Sn, test_Sp, mcc, roc_auc))
    print("confusion matrix:\n"+str (confusion_matrix(test_label, predict_label)))
    #plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i , roc_auc))
    print('---------------------------------------------------')
    
    print('test_Accuracy:'+str(format(test_acc, '.4f'))+', test_Sn:'+str(format(test_Sn, '.4f'))+', test_Sp:'+str(format(test_Sp, '.4f'))
                                +', mcc:'+str(format(mcc, '.4f'))+', roc_auc'+str(format(roc_auc, '.4f')), file=doc)
    print("confusion matrix:\n"+str (confusion_matrix(test_label, predict_label)), file=doc)
    print('---------------------------------------------------', file=doc)            

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

print('Final_test_Accuracy:' + str(format(Final_test_acc, '.4f')), file=doc)   
print('Final_test_Sn:' + str(format(Final_test_Sn, '.4f')), file=doc) 
print('Final_test_Sp:' + str(format(Final_test_Sp, '.4f')), file=doc) 
print('Final_mcc:' + str(format(Final_mcc, '.4f')), file=doc) 
print('Final_AUC:' + str(format(Final_auc, '.4f')), file=doc)

   
