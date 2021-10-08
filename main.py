import pandas as pd
df=pd.read_csv('train.csv')
df=df.dropna()
X=df.drop('label',axis=1)
y=df['label']
import tensorflow as ts
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import nltk
from nltk.corpus import stopwords
import numpy as np
vocab_size=5000 #size of vocabulary
messages = X.copy()
messages.reset_index(inplace=True)
#text preprocessing
# lemmatizer declaration
from nltk.stem import WordNetLemmatizer
import re
lemmatizer = WordNetLemmatizer()
corpus = []
#stopword removal and lemmatization
for i in range(0,len(messages)):
    rev = re.sub('[^a-zA-Z]',' ',messages['title'][i])
    rev = rev.lower()
    rev= rev.split()
    rev = [lemmatizer.lemmatize(word) for word in rev if not word in stopwords.words('english')]
    rev = ' '.join(rev)
    corpus.append(rev)
#visualizing the tweets
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
word2vec_model = Word2Vec(messages, min_count=1)
X = word2vec_model[word2vec_model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
emb_words = list(word2vec_model.wv.vocab)
for i, word in enumerate(emb_words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
#one hot conversion
onehot_rep = [one_hot(words,vocab_size) for words in corpus]
#padding the sentences to get equal lengths
sentence_length = 20
embedded_docs = pad_sequences(onehot_rep,padding='pre',maxlen=sentence_length)
embedding_vector_features = 20
print(embedded_docs)
#lstm model creation
embedding_vector_features=50
model=Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=sentence_length))
model.add(LSTM(100)) #lstm layer with 100 neurons
model.add(Dense(1,activation='sigmoid')) #sigmoid for prob clac
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#summary of model
print(model.summary())
#training the model
X_final=np.array(embedded_docs)
y_final=np.array(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
#fitting the model
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
from tensorflow.keras.layers import Dropout
# Creating dropout model
embedding_vector_features=50
model=Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=sentence_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
y_pred = (model.predict(X_test) > 0.5).astype("int32")
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
