import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.multiclass import OneVsOneClassifier , OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix , recall_score , precision_score  , f1_score , precision_recall_curve ,ConfusionMatrixDisplay , accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import GridSearchCV
mnist = fetch_openml('mnist_784' , as_frame = False)
#print(type(mnist)) #<class 'sklearn.utils._bunch.Bunch'>
x : np.ndarray = mnist.data 
y : np.ndarray = mnist.target

def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image , cmap='binary')
    plt.axis('off')

def plot_digits(data):
    plt.figure(figsize=(9,9))
    for idx , image_data in enumerate(data):
        plt.subplot(5 , 10 , idx + 1 )
        plot_digit(image_data)
    plt.subplots_adjust(wspace=0 , hspace=0)

# splite data to  Xtrain , Ytrain , xtest , ytestu
x_train , y_train , x_test , y_test = x[:60000] , y[:60000] , x[60000:] , y[60000:]
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
some_digit = x_train[0]

#draw_data
'''
[In] plot_digit(x_train[0])
[Out] figure have data in this case will draw 5
_____

[In] plot_digits(x_train[:50])
[Out] figure have 50 photos # you can increase but in average use 50 : 80 photo
'''

# binary classifier
def SGD_clasifier(): # extra func
    sgd_cls = SGDClassifier(random_state = 42)                               
    sgd_cls.fit(x_train , y_train_5) # to fit the model                      
    print(sgd_cls.get_params()) # to understand his hyperparam
    sgd_cross = cross_val_score(sgd_cls , x_train , y_train_5 , cv=3 , scoring = 'accuracy') # as we use that before and our score = accuracy (دقه)
    sgd_cross_pre = cross_val_predict(sgd_cls , x_train , y_train_5 , cv=3) # like cross_val_score but this will not give you scoring  , will give predictions
    print(confusion_mtrx := confusion_matrix(y_train_5 , sgd_cross_pre)) # give you confusion_matrix will give you 2D arrray [top left {TN} , top rghit {FP} , lower left {FN} , lower right {TP}]
    print(precision_score(y_train_5 , sgd_cross_pre)) # return precision
    print(recall_score(y_train_5 , sgd_cross_pre)) # return recall
    print(f1_score(y_train_5 , sgd_cross_pre)) 
    scores = sgd_cls.decision_function(some_digit) # return threhold for each element
    
    therhold = 3000 # suppose threhold 

    y_socres  = cross_val_predict(sgd_cls,x_train  , y_train_5 , cv=3 ,method = 'decision_function')  # will give you threhold for each element
    precision , recall , therholds = precision_recall_curve(y_train_5 , y_socres) # will give precision , recall , threhold

    def draw_recall_precision_curve():
        plt.plot(therholds , precision[:-1] , label = 'precision' , linewidth = 2)
        plt.plot(therholds , recall[:-1] , label = 'recall' , linewidth = 2)
        plt.vlines(therhold , 0 ,1.0 ,'k' ,'dotted' , label='threhold')
        plt.legend()
        plt.grid()
        plt.axis([-50000 , 50000 ,0 ,1])
        plt.show()
    def draw_recall_against_directly_precision():
        idx = (therholds >= therhold).argmax()
        plt.plot(recall , precision , label = 'precision_against_recall' , linewidth = 2)
        plt.plot([recall[idx]] , [precision[idx]] , 'ko' , label = 'point at threhold')
        plt.grid()
        plt.vlines(therhold , 0 ,1.0 ,'k' ,'dotted' , label='threhold')
        plt.legend()
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.axis([0,1,0,1])
        plt.show()
    
    idx_for_90_precision = (precision >= 0.90).argmax()   # get first index will make precision == 90%
    threhold_for_90_pre = therholds[idx_for_90_precision] # get this threhold that will make the precision == 90%
    y_train_with_90_pre = (y_socres >= threhold_for_90_pre) # will down or up the threhold as you write ( in this case will up the threhold to make the precision more high)

def finall_binary_classifier():
    sgd_cls = SGDClassifier(random_state = 42)                               
    sgd_cls.fit(x_train , y_train_5) # to fit the model   
    print(y_train_5[:20]) # first 20 index
    print(sgd_cls.predict(some_digit)) # predict some data                 
    #sgd_cross_pre = cross_val_predict(sgd_cls , x_train , y_train_5 , cv=3) # like cross_val_score but this will not give you scoring  , will give predictions

'''
[In] finall_binary_classifier()
[Out] [True , False ,.........]
'''
def svc_classifier_multi():
    svc_classifirer = SVC(random_state = 42)
    svc_classifirer.fit(x_train[:3000] , y_train[:3000]) # feed the classifier
    print(svc_classifirer.predict([some_digit])) # will give us a perdictons
    #here we will get raito of all classifiers has been  feeded
    '''
    score_digit = svc_classifirer.decision_function([some_digit[0]]) # return raito of confidence will return raito from all class and get high confidence ratio and classed it on this base
    class_id = score_digit.argmax() # will get index of high raito
    print(svc_classifirer.classes_) # will return all values be classed
    print(svc_classifirer.classes_[class_id]) # return the classed obj
    '''

'''
[In] svc_classifier_multi()
[Out] 5 as we expected
'''
def control_as_we_want_ovo_or_ovr():  # extra_func
    one_vs_rest = OneVsRestClassifier(SVC(random_state = 42)) # this method will bulid 10 models and get most raito in all predict and classed it on this base   
    one_vs_one = OneVsOneClassifier(SVC(random_state = 42)) # this method will bulid N * (N-1)/2 in our case will train 45 models 10 *(10-1) /2
    one_vs_rest.fit(x_train[:3000] , y_train[:3000])
    one_vs_one.fit(x_train[:3000] , y_train[:3000])
    #to choose who you will pick you should measure who better by cross_val_score method like:
    '''
    print('OvR',cross_val_score(one_vs_rest , x_train[:3000] , y_train[:3000] , cv=3 , scoring='accuracy'))
    print('OvO',cross_val_score(one_vs_one , x_train[:3000  ] , y_train[:3000] , cv=3 , scoring='accuracy'))
    '''

def sgd_with_error_analysis(): # extra_func

    # here will will analysis where our model make mistake
    sgd_cls = SGDClassifier(random_state=42) # or any classifier
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x_train.astype('float64'))
    y_train_pred = cross_val_predict(sgd_cls , scaled_x , y_train , cv=3 )
    #print(f'without_scale = {cross_val_score(sgd_cls , x_train , y_train , scoring ="accuracy" )}') # [0.88083333  , 0.88325 , 0.88116667 , 0.86625 , 0.8875]
    #print(f'without_scale = {cross_val_score(sgd_cls , scaled_x , y_train , scoring ="accuracy" , cv=3 )}') # [0.89733333 , 0.88725 , 0.89583333 , 0.89233333 , 0.90516667]
    sample_weight = (y_train_pred != y_train)
    print(sample_weight)
    plt.figure(figsize=(12 , 10))
    ConfusionMatrixDisplay.from_predictions(y_train , y_train_pred) # get raito of all true labels in main digonal
    ConfusionMatrixDisplay.from_predictions(y_train , y_train_pred , normalize='true' , values_format='.0%')
    ConfusionMatrixDisplay.from_predictions(y_train , y_train_pred , normalize='true', sample_weight=sample_weight , values_format='.0%')
    plt.show()  

def try_to_get_more_accuracy(): # extra_func * important cuz this will give us best estimator
    #here we will promote KNN classifier to get 97% accuracy 
    num_of_digits = 60000 # as you want but take care about you computer (i suggest put 15000 then put more untill you computer be slow)
    param_grid = {
        'n_neighbors':[3,4,6],
        'weights':['uniform', 'distance']
    }
    cls = KNeighborsClassifier()
    grid_serach_ = GridSearchCV(cls ,cv=3 , param_grid=param_grid , scoring='accuracy')
    grid_serach_.fit(x_train[:num_of_digits] , y_train[:num_of_digits])
    print(grid_serach_.best_score_)
    print(grid_serach_.best_params_)
    # you can get best estimator easy do this :
    #best_estimator = grid_serach_.best_estimator_
    #best_estimator.fit()...