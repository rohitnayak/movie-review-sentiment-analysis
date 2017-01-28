
# coding: utf-8

# In[1]:

import pyprind
import pandas as pd
import os
pbar = pyprind.ProgBar(25000)
labels = {'pos': 1, 'neg': 0}
df=pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos','neg'):
        path = './aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'rt', encoding="utf8") as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
                pbar.update()
df.columns = ['review','sentiment']


# In[2]:

import numpy as np
np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))


# In[3]:

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+',' ',text.lower()) + ' '.join(emoticons).replace('-','')
    return text


# In[4]:

df['review']=df['review'].apply(preprocessor)


# In[ ]:

np.set_printoptions(precision=2)
X_train=df.loc[:1000,'review'].values
y_train=df.loc[:1000,'sentiment'].values

X_test=df.loc[1000:2000,'review'].values
y_test=df.loc[1000:2000,'sentiment'].values

import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
stop = stopwords.words('English')
from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()

def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

count = CountVectorizer()
tfidf = TfidfVectorizer(strip_accents=False, preprocessor=None, lowercase=False)

param_grid = [
    {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__penalty': ['l1','l2'],
        'clf__C': [1.0, 10.0, 100.0]        
    },
    {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__penalty': ['l1','l2'],
        'clf__C': [1.0, 10.0, 100.0]        
    }
    ]
best_param_grid = [
    {
        'vect__ngram_range': [(1,1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer],
        'clf__penalty': ['l2'],
        'clf__C': [10.0]        
    }
]
lr = LogisticRegression(random_state=0)
lr_tfidf = Pipeline([('vect', tfidf), ('clf', lr)])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', n_jobs=1, cv=5, verbose=1)

# In[ ]:

gs_lr_tfidf.fit(X_train, y_train)


# In[ ]:

print("CV Accuracy: %0.3f" % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_

print("Test Accuracy: %0.3f" % clf.score(X_test, y_test))

# In[ ]:



