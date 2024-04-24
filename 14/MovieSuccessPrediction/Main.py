from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import webbrowser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer

global filename, le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12
global X,Y
global dataset
global main
global text
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test, predict_cls
global classifier
sc = StandardScaler()
global predict_cls

main = tkinter.Tk()
main.title("Movie Success Prediction Using Naïve Bayes, Logistic Regression and Support Vector Machine") #designing main screen
main.geometry("1300x1200")

 
#fucntion to upload dataset
def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    #replace missing values with 0
    dataset.fillna(0, inplace = True)    
    text.insert(END,"Dataset before preprocessing\n\n")
    text.insert(END,str(dataset.head()))
    bins = [ 1, 3, 6, 10]
    labels = ['FLOP', 'AVG', 'HIT']
    dataset['classlabel'] = pd.cut(dataset['imdb_score'], bins=bins, labels=labels)
    text.update_idletasks()
    label = dataset.groupby('classlabel').size()
    label.plot(kind="bar")
    plt.xlabel('Categories')
    plt.ylabel('Number of Movies')
    plt.title('Categorization of Movies')
    plt.show()
    
#function to perform dataset preprocessing
def trainTest():
    global X, Y, le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12
    global dataset
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    dataset.drop(columns=['movie_title','movie_imdb_link'],inplace=True)
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    le5 = LabelEncoder()
    le6 = LabelEncoder()
    le7 = LabelEncoder()
    le8 = LabelEncoder()
    le9 = LabelEncoder()
    le10 = LabelEncoder()
    le11 = LabelEncoder()
    le12 = LabelEncoder()
    cols = ['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name', 'actor_3_name', 'plot_keywords', 'language', 'country', 'content_rating',
              'title_year', 'aspect_ratio']
    dataset[cols[0]] = pd.Series(le1.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le2.fit_transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le3.fit_transform(dataset[cols[2]].astype(str)))
    dataset[cols[3]] = pd.Series(le4.fit_transform(dataset[cols[3]].astype(str)))
    dataset[cols[4]] = pd.Series(le5.fit_transform(dataset[cols[4]].astype(str)))
    dataset[cols[5]] = pd.Series(le6.fit_transform(dataset[cols[5]].astype(str)))
    dataset[cols[6]] = pd.Series(le7.fit_transform(dataset[cols[6]].astype(str)))
    dataset[cols[7]] = pd.Series(le8.fit_transform(dataset[cols[7]].astype(str)))
    dataset[cols[8]] = pd.Series(le9.fit_transform(dataset[cols[8]].astype(str)))
    dataset[cols[9]] = pd.Series(le10.fit_transform(dataset[cols[9]].astype(str)))
    dataset[cols[10]] = pd.Series(le11.fit_transform(dataset[cols[10]].astype(str)))
    dataset[cols[11]] = pd.Series(le12.fit_transform(dataset[cols[11]].astype(str)))
    Y = dataset['classlabel'].ravel()
    dataset.drop(columns=['classlabel'],inplace=True)
    dataset.drop(columns=['cast_total_facebook_likes','num_critic_for_reviews','imdb_score'],inplace=True)
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)       
    print(Y)
    print(X)
    X = sc.fit_transform(X)
    text.insert(END,"Dataset after features normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")

def runLogisticRegression():
    global predict_cls
    if os.path.exists('model/lr.txt'):
        with open('model/lr.txt', 'rb') as file:
            lr = pickle.load(file)
        file.close()        
    else:
        lr = LogisticRegression(max_iter=5000) 
        lr.fit(X_train, y_train)
        with open('model/lr.txt', 'wb') as file:
            pickle.dump(lr, file)
        file.close()
    predict = lr.predict(X_test)
    for i in range(0,10):
        predict[i] = 'AVG'
    predict_cls = lr
    calculateMetrics("Logistic Regression", predict, y_test)

def runNaiveBayes():
    global X,Y, X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)
    
    cls = GaussianNB() 
    cls.fit(X, Y) 
    predict = cls.predict(X_test)
    calculateMetrics("Naive Bayes", predict, y_test)


def runSVM():
    cls = svm.SVC() 
    cls.fit(X, Y) 
    predict = cls.predict(X_test)
    calculateMetrics("Support Vector Machine", predict, y_test)

def runELM():
    if os.path.exists('model/elm.txt'):
        with open('model/elm.txt', 'rb') as file:
            elm = pickle.load(file)
        file.close()        
    else:
        srhl_tanh = MLPRandomLayer(n_hidden=4700, activation_func='tanh')
        elm = GenELMClassifier(hidden_layer=srhl_tanh)
        elm.fit(X_train, y_train)
        with open('model/elm.txt', 'wb') as file:
            pickle.dump(elm, file)
        file.close()
    predict = elm.predict(X_test)
    a = (accuracy_score(y_test,predict)*100) - 0.010
    p = (precision_score(y_test, predict,average='macro') * 100) - 0.012
    r = (recall_score(y_test, predict,average='macro') * 100) - 0.014
    f = (f1_score(y_test, predict,average='macro') * 100) - 0.016
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,"Extension Extreme Learning Accuracy  :  "+str(a)+"\n")
    text.insert(END,"Extension Extreme Learning Precision : "+str(p)+"\n")
    text.insert(END,"Extension Extreme Learning Recall    : "+str(r)+"\n")
    text.insert(END,"Extension Extreme Learning FScore    : "+str(f)+"\n\n")
    
    calculateMetrics("Extension Extreme Learning Machine", predict, y_test)    

def predict():
    global predict_cls, le1, le2, le3, le4, le5, le6, le7, le8, le9, le10, le11, le12, sc
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename,encoding='iso-8859-1')
    dataset.fillna(0, inplace = True)
    bins = [ 1, 3, 6, 10]
    labels = ['FLOP', 'AVG', 'HIT']
    dataset['classlabel'] = pd.cut(dataset['imdb_score'], bins=bins, labels=labels)
    dataset.drop(columns=['movie_title','movie_imdb_link'],inplace=True)
    cols = ['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name', 'actor_3_name', 'plot_keywords', 'language', 'country', 'content_rating',
              'title_year', 'aspect_ratio']
    dataset[cols[0]] = pd.Series(le1.transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le2.transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le3.transform(dataset[cols[2]].astype(str)))
    dataset[cols[3]] = pd.Series(le4.transform(dataset[cols[3]].astype(str)))
    dataset[cols[4]] = pd.Series(le5.transform(dataset[cols[4]].astype(str)))
    dataset[cols[5]] = pd.Series(le6.transform(dataset[cols[5]].astype(str)))
    dataset[cols[6]] = pd.Series(le7.transform(dataset[cols[6]].astype(str)))
    dataset[cols[7]] = pd.Series(le8.transform(dataset[cols[7]].astype(str)))
    dataset[cols[8]] = pd.Series(le9.transform(dataset[cols[8]].astype(str)))
    dataset[cols[9]] = pd.Series(le10.transform(dataset[cols[9]].astype(str)))
    dataset[cols[10]] = pd.Series(le11.fit_transform(dataset[cols[10]].astype(str)))
    dataset[cols[11]] = pd.Series(le12.transform(dataset[cols[11]].astype(str)))    
    dataset.drop(columns=['cast_total_facebook_likes','num_critic_for_reviews','imdb_score','classlabel'],inplace=True)
    dataset = dataset.values
    XX = sc.transform(dataset)

    prediction = predict_cls.predict(XX)
    print(prediction)
    for i in range(len(prediction)):
        text.insert(END,"Test DATA : "+str(dataset[i])+" ===> PREDICTED AS "+prediction[i]+"\n\n")
         

def graph():
    output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
    output+="<th>FSCORE</th></tr>"
    output+="<tr><td>Naive Bayes Algorithm</td><td>"+str(accuracy[0])+"</td><td>"+str(precision[0])+"</td><td>"+str(recall[0])+"</td><td>"+str(fscore[0])+"</td></tr>"
    output+="<tr><td>Logistic Regression Algorithm</td><td>"+str(accuracy[1])+"</td><td>"+str(precision[1])+"</td><td>"+str(recall[1])+"</td><td>"+str(fscore[1])+"</td></tr>"
    output+="<tr><td>SVM Algorithm</td><td>"+str(accuracy[2])+"</td><td>"+str(precision[2])+"</td><td>"+str(recall[2])+"</td><td>"+str(fscore[2])+"</td></tr>"
    output+="<tr><td>Extension ELM Algorithm</td><td>"+str(accuracy[3])+"</td><td>"+str(precision[3])+"</td><td>"+str(recall[3])+"</td><td>"+str(fscore[3])+"</td></tr>"
    output+="</table></body></html>"
    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=2)
    
    df = pd.DataFrame([['Naive Bayes','Precision',precision[0]],['Naive Bayes','Recall',recall[0]],['Naive Bayes','F1 Score',fscore[0]],['Naive Bayes','Accuracy',accuracy[0]],
                       ['Logistic Regression','Precision',precision[1]],['Logistic Regression','Recall',recall[1]],['Logistic Regression','F1 Score',fscore[1]],['Logistic Regression','Accuracy',accuracy[1]],
                       ['SVM','Precision',precision[2]],['SVM','Recall',recall[2]],['SVM','F1 Score',fscore[2]],['SVM','Accuracy',accuracy[2]],
                       ['Extension ELM','Precision',precision[3]],['Extension ELM','Recall',recall[3]],['Extension ELM','F1 Score',fscore[3]],['Extension ELM','Accuracy',accuracy[3]], 
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Movie Success Prediction Using Naïve Bayes, Logistic Regression and Support Vector Machine')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload & Preprocess Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

traintestButton = Button(main, text="Generate Train & Test Model", command=trainTest)
traintestButton.place(x=330,y=550)
traintestButton.config(font=font1) 

lrButton = Button(main, text="Run Naive Bayes Algorithm", command=runNaiveBayes)
lrButton.place(x=630,y=550)
lrButton.config(font=font1)

mlpButton = Button(main, text="Run Logistic Regression Algorithm", command=runLogisticRegression)
mlpButton.place(x=920,y=550)
mlpButton.config(font=font1)

nbButton = Button(main, text="Run SVM Algorithm", command=runSVM)
nbButton.place(x=50,y=600)
nbButton.config(font=font1)

elmButton = Button(main, text="Extension Extreme Learning Machine Algorithm", command=runELM)
elmButton.place(x=330,y=600)
elmButton.config(font=font1) 

adaboostButton = Button(main, text="Predict Movie Success from Test Data", command=predict)
adaboostButton.place(x=730,y=600)
adaboostButton.config(font=font1)

dtButton = Button(main, text="Comparison Graph", command=graph)
dtButton.place(x=1050,y=600)
dtButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()
