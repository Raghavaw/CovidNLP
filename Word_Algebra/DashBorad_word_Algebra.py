#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import gensim 
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
import nltk
import pandas as pd
from wordcloud import WordCloud, STOPWORDS 
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gensim 
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
import nltk
from gensim.models import Word2Vec


# In[2]:


model = Word2Vec.load("word2vec_whoData.model")


# In[3]:


pos = ['Pneumonia','Covid']##### tab for postive 
neg = [] ###### tab for negative 
equation = "+".join(pos) +"-" +"-".join(neg)
print(equation)


# In[4]:


res=model.most_similar(positive= [x.lower() for x in pos],negative= [x.lower() for x in neg], topn=3, restrict_vocab=None, indexer=None)


# In[5]:


print('top 3 most most probable answers are [{},{},{}]'.format(res[0][0],res[1][0],res[2][0]))


# In[ ]:




