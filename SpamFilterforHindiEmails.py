
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, preprocessing
from sklearn.svm import SVC,NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score , GridSearchCV,train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from IPython.display import Image
import re
import pickle
import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')


# In[2]:

from keras.layers import Embedding, Dense, LSTM , Dropout
from keras.models import Sequential
import seaborn as sns; sns.set()


# In[13]:

train = pd.read_excel("hindi_spam_msg.xlsx")
print(train)


# In[14]:

print("Frequency of spam and ham E-Mails:\n",train['type'].value_counts())
count_Class=pd.value_counts(train["type"], sort= True)
count_Class.plot(kind = 'pie',  autopct='%1.0f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()
count_Class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()


# # REMOVING PUNCTUATIONS , DIGITS & ENGLISH LETTERS

# In[15]:

def purify(doc):
    punctuations = '''!@#$£%^&*()"':-+_=“”/?><|।.,\{}[]'''
    digits = '''1234567890०१२३४५६७८९'''
    english = '''abcdefghijklmnopqrstuvwxyzABCCDEFGHIJKLMNOPQRSTUVWXYZ'''
    data_no_punc = ""
    for char in doc:
        if(char not in punctuations and char not in digits and char not in english):
            data_no_punc = data_no_punc + char
    return data_no_punc
i=0
for e in train['text']:
    train['text'][i]=''.join(list(map(purify,e)))
    train['text'][i]=train['text'][i].split()
    i=i+1


# # REMOVING STOP-WORDS

# In[16]:

# stopwords = open('stopwords.txt','r')
with open('stopwords.txt') as f:
    stopwords = f.read()
# stwords=stopwords.read()
words = re.sub('\n',' ',stopwords)
word = words.split()

def rem_stop(doc):
    clearW = [wor for wor in doc if wor not in word]
    return clearW

train['text']=list(map(rem_stop,train['text']))


# # REMOVING SUFFIXES

# In[17]:

suffixes = {
    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
    2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती","एं", "ता", "ाँ", "ां", "ों", "ें","ओं","ाना",
        "ाऊ","लु","ाव","ीय","हट"],
    3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं",
        "ाएं", "ुओं", "ुएं", "ुआं","दार","िया","हार","ावट","ोला","कार","ान","ौटी","ैया","ेरा","ोड़ा"],
    4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं",
        "ताएं", "ियाँ", "ियों", "ियां","वाला","िक"],
    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां","क्कड़"]
}

def hi_stem(lst):
    txt = []
    for word in lst: 
        flag = 0
        for L in 5, 4, 3, 2, 1:
            if len(word) > L + 1:
                for suf in suffixes[L]:
                    if word.endswith(suf):
                        word = word[:-L]
                        flag = 1
                        break
                if flag == 1:
                    break
        txt.append(word)      
    return txt

email = []
email = (list(map(hi_stem,train['text'])))
train['text']= email

for  i in range(0,train['text'].shape[0]):
    train['text'][i] = ' '.join(train['text'][i])

print(train)


# In[20]:

# train_x,valid_x,train_y,valid_y = train_test_split(train['text'],train['type'],test_size=0.2,stratify=train['type'].values,random_state=2)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train['type'])
# valid_y = encoder.fit_transform(valid_y)


# In[21]:

# tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=10000)
# tfidf_vect.fit(train['text'])
# x_tfidf = tfidf_vect.transform(train.text)
# xtrain_tfidf =  tfidf_vect.transform(train_x)
# xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=2000)
train_tfid = tfidf_vect_ngram.fit_transform(train['text'])
# x_tfidf_ngram = tfidf_vect_ngram.transform(train.text)
# xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
# xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)


# In[22]:

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y),confusion_matrix(valid_y, predictions)


# In[11]:

# Naive Bayes on Ngram Level TF IDF Vectors
NBAccuracy = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: ", NBAccuracy)


# In[12]:

# SVM SVC on Ngram Level TF IDF Vectors
SVCAccuracy = train_model(SVC(kernel='linear'),xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVC, N-gram TF-IDF: ", SVCAccuracy)


# In[13]:

# SVM NuSVC on Ngram Level TF IDF Vectors
NuSVCAccuracy = train_model(NuSVC(kernel='linear',probability=True),xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVC, N-gram TF-IDF: ", NuSVCAccuracy)


# In[14]:

# RF on n-grams Level TF IDF Vectors
RFAccuracy = train_model(RandomForestClassifier(n_estimators=250,min_samples_split=10), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("RF, N-gram TF-IDF: ", RFAccuracy)


# In[15]:

# GB on n-grams Level TF IDF Vectors
GBAccuracy = train_model(GradientBoostingClassifier(n_estimators=250,min_samples_split=200), xtrain_tfidf_ngram.tocsc(), train_y, xvalid_tfidf_ngram.tocsc())
print("Gradient Boost, N-Grams Vectors: ", GBAccuracy)


# In[16]:

# KNN on n-grams Level TF IDF Vectors
KNNAccuracy = train_model(KNeighborsClassifier(n_neighbors=10),xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("KNN, N-gram TF-IDF: ", KNNAccuracy)


# In[17]:

# Stacking using all above algos
Stack_Accuracy = train_model(VotingClassifier(estimators=[('SVC', SVC(kernel='linear',probability=True).fit(xtrain_tfidf_ngram, train_y)), ('RF', RandomForestClassifier(n_estimators=250,min_samples_split=3).fit(xtrain_tfidf_ngram, train_y)),('KNN',KNeighborsClassifier(n_neighbors=10).fit(xtrain_tfidf_ngram, train_y)),('NB',naive_bayes.BernoulliNB().fit(xtrain_tfidf_ngram, train_y)),('NuSVC',NuSVC(kernel='linear',probability=True).fit(xtrain_tfidf_ngram, train_y)),('GB',GradientBoostingClassifier(n_estimators=250,min_samples_split=200).fit(xtrain_tfidf_ngram, train_y))], voting='soft',weights=[6,5,4,3,2,1]), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("Stacking, N-gram TF-IDF: ", Stack_Accuracy)


# In[23]:

NBModel = naive_bayes.BernoulliNB().fit(x_tfidf_ngram,train['type'])
SVCModel = SVC(kernel='linear',probability=True).fit(x_tfidf_ngram,train['type'])
NuSVCModel = NuSVC(kernel='linear',probability=True).fit(x_tfidf_ngram,train['type'])
RFModel = RandomForestClassifier(n_estimators=250,min_samples_split=3).fit(x_tfidf_ngram,train['type'])
GBModel = GradientBoostingClassifier(n_estimators=250,min_samples_split=200).fit(x_tfidf_ngram,train['type'])
KNNModel = KNeighborsClassifier(n_neighbors=10).fit(x_tfidf_ngram,train['type'])
StackModel = VotingClassifier(estimators=[('SVC', SVCModel), ('RF', RFModel),('KNN',KNNModel),('NB',NBModel),('NuSVC',NuSVCModel),('GB',GBModel)], voting='soft',weights=[6,5,4,3,2,1]).fit(x_tfidf_ngram,train['type'])


# In[19]:

f1 = "NB.sav"
pickle.dump(NBModel,open(f1,'wb'))

f2 = "SVC.sav"
pickle.dump(SVCModel,open(f2,'wb'))

f3 = "NuSC.sav"
pickle.dump(NuSVCModel,open(f3,'wb'))

f4 = "RF.sav"
pickle.dump(RFModel,open(f4,'wb'))

f5 = "GB.sav"
pickle.dump(GBModel,open(f5,'wb'))

f6 = "KNN.sav"
pickle.dump(GBModel,open(f6,'wb'))

f7 = "Stack.sav"
pickle.dump(GBModel,open(f7,'wb'))


# In[20]:

with open("Stack.sav",'rb') as f17:
    load_stack = pickle.load(f17)


# In[24]:

str = 'आपने लॉटरी जीती है'
str=purify(str)
print(str)
str=str.split()
str=rem_stop(str)
print(str)
str = hi_stem(str)
print(str)
str = ' '.join(str)
str

l = []
l.append(str)
str_tfidf_ngram = tfidf_vect_ngram.transform(l)

StackModel.predict(str_tfidf_ngram)




