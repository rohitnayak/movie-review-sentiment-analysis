

import numpy as np

import re
from nltk.corpus import stopwords
stop = stopwords.words('english')

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+',' ',text.lower()) + ' '.join(emoticons).replace('-','')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path): 
    with open(path, 'r') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [],[]
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',#norm='l2',ngram_range=(2,2),
                            n_features=2**21, preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

doc_stream = stream_docs(path='./movie_data.csv')

import pyprind
num_batches = 45
pbar = pyprind.ProgBar(num_batches)
train_batch_size = 1000
test_batch_size = 5000

classes = np.array([0, 1])
for _ in range(num_batches):
    X_train, y_train = get_minibatch(doc_stream, size=train_batch_size)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=test_batch_size)
X_test = vect.transform(X_test)

print("Test Accuracy: %0.3f" % clf.score(X_test, y_test))

clf = clf.partial_fit(X_test, y_test, classes=classes)

import pickle, os

dest = os.path.join('movieclassifier', 'pickled_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop,
    open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
    protocol=4)    
pickle.dump(clf,
    open(os.path.join(dest, 'classifier.pkl'), 'wb'),
    protocol=4) 


#https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html   

