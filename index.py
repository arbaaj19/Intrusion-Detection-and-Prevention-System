from tkinter import *
from PIL import ImageTk,Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lm
from sklearn import svm, metrics
import joblib
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

root = Tk()
root.geometry('1100x600')
root.title("IDS")

def hund_val():
    six_op.grid_remove()
    six_op1.grid_remove()
    label_blank = Label(root, text="  ", padx="20", pady="20")
    label_blank.grid(row=2,column=0)
    global label_info
    label_info = Label(root, text="Enter record number you want to compare (0 to 494020)", padx="20", pady="20")
    label_info.config(font=("Courier",14))
    label_info.grid(row=3,column=0,columnspan=2)
    global value
    value = Entry(root, width=40, borderwidth=5)
    value.grid(row=4,column=0)
    global btn_done
    btn_done = Button(root, text="Done", command=lambda: hund_exe(value.get()))
    btn_done.grid(row=4,column=1)
    
def hund_exe(val):
    label_info.grid_remove()
    value.grid_remove()
    btn_done.grid_remove()
    row_no = int(val)
    ##Import dataset
    dataset = pd.read_csv('C:\Python\BE\Version 3 - 28-02-2020(WEENGS)\kddcup99.csv',low_memory=False)

    ##Import Trained Regression Model
    model = joblib.load('model.pkl')
    
    ##dictionary
    dict = ['Dos - Back','u2r-buffer_overflow ','r2l-ftp_write','r2l-guess_passwd','r2l-imap',
        'probe-ipsweep','dos-land','u2r-loadmodule','r2l-multihop','dos-neptune','probe-nmap',
        'u2r-perl','r2l-phf','dos-pod','probe-portsweep','u2r-rootkit','probe-satan',
        'dos-smurf','r2l-spy','dos-teardrop','r2l-warezclient','r2l-warezmaster','normal-normal' ]
    y = dataset.label
    x = np.array(dataset.drop(['flag'],axis=1))

    output = model.predict(np.reshape(x[row_no,:],(1,-1)))
    output = int(round(output[0]))-1
    original = y[row_no]-1
    p = "Predictions \t"+dict[output]+"\nOriginal\t"+dict[original]
    op = Label(root, text=p, padx="50", pady="60")
    op.config(font=("Courier",24))
    op.grid(row=5,column=0,columnspan=2)
    
    root1 = Tk()
    root1.title("Attributes")
    disp = PrettyTable()
    attr = ["Attribute Name","duration","protocol_type","service","src_bytes","dst_bytes","land","wrong_fragment",
                        "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
                        "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
                        "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
                        "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
                        "dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
                        "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]

    disp.field_names = ["Attribute", "Tested Data"]
    for i in range (0, 41):
        disp.add_row([attr[i],x[row_no,i]])

    p1 = disp
    op1 = Label(root1, text=p1)
    op1.config(font=("Courier",10))
    op1.grid(row=5,column=0,columnspan=2)

def sixty():
    #importing the dataset
    dataset = pd.read_csv('kddcup.data_10_percent_corrected')

    #change Multi-class to binary-class
    dataset['normal.'] = dataset['normal.'].replace(['back.', 'buffer_overflow.', 'ftp_write.', 'guess_passwd.', 'imap.', 'ipsweep.', 'land.', 'loadmodule.', 'multihop.', 'neptune.', 'nmap.', 'perl.', 'phf.', 'pod.', 'portsweep.', 'rootkit.', 'satan.', 'smurf.', 'spy.', 'teardrop.', 'warezclient.', 'warezmaster.'], 'attack')

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 41].values
    #encoding categorical data
    labelencoder_x_1 = LabelEncoder()
    labelencoder_x_2 = LabelEncoder()
    labelencoder_x_3 = LabelEncoder()
    x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
    x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
    x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

    #splitting the dataset into the training set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)

    #feature scaling
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)

    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
    accuracies.mean()
    accuracies.std()

    #the performance of the classification model
    precision = cm[1,1]/(cm[1,0]+cm[1,1])
    p = "The accuracy is: "+ str((cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]))
    p1 = "The precision is: "+ str(precision)
    global six_op
    six_op = Label(root, text=p, padx="20", pady="20")
    six_op.config(font=("Courier",24))
    six_op.grid(row=5,column=0,columnspan=2)
    global six_op1
    six_op1 = Label(root, text=p1, padx="20", pady="20")
    six_op1.config(font=("Courier",24))
    six_op1.grid(row=6,column=0,columnspan=2)
##    recall = cm[1,1]/(cm[0,1]+cm[1,1])
##    print("Recall is : "+ str(recall))
##    print("False Positive rate: "+ str(cm[1,0]/(cm[0,0]+cm[1,0])))
    
##    print("F-measure is: "+ str(2*((precision*recall)/(precision+recall))))
##    from math import log
##    print("Entropy is: "+ str(-precision*log(precision)))

def page1():
    btn_next.grid_remove()
    btn1 = Button(root, text="Train 60% Test 40%", borderwidth = 5,padx="50", pady="20", fg = "white", bg = "black", command=sixty)
    btn1.grid(row=1,column=0)
    btn2 = Button(root, text="Train 100% Test 1 record", borderwidth = 5,padx="60", pady="20", fg = "white", bg = "black", command=hund_val)
    btn2.grid(row=1,column=1)

title = Label(root, text="Intrusion Detection System", padx="50", pady="80",)
title.config(font=("Courier",44))
title.grid(row=0,column=0,columnspan=2)

btn_next = Button(root, text = "Next", borderwidth = 5,padx="50", pady="20", fg = "white", bg = "black", command=page1)
btn_next.grid(row=1,column=0,columnspan=2)



root.mainloop()
