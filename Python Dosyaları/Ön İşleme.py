import pandas as pd
import numpy as np


import re
import nltk as nlp
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#%% Veri Kümelerinin İmport Edilmesi

data1 = Traincsv
data2 = Testcsv
data3 = Validcsv
data_list = pd.concat([data1,data2,data3], axis = 0)

data_list.dropna(inplace = True)
text_list = []

x = data_list.text
y = data_list.label
#%%

for i in x:
    i = re.sub("[^a-zA-Z]" ," ",i)
    i = i.lower()
    i = nlp.word_tokenize(i)
    i = [word for word in i if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    i = [lemma.lemmatize(word) for word in i]
    i = " ".join(i)
    text_list.append(i)
 
