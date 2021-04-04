#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lsa import LsaSummarizer #We're choosing Luhn, other algorithms are also built in
from nltk.tokenize import RegexpTokenizer


# In[21]:


df = pd.read_csv('/WHO_new.csv', encoding = 'utf-8-sig')


# In[44]:


TB=df[df.Title.str.contains('tuberculosis', regex= True, na=False,case=False)]
Cardio=df[df.Title.str.contains('cardio', regex= True, na=False,case=False)]
HIV=df[df.Title.str.contains('HIV', regex= True, na=False,case=False)]
Maternal = df[df.Title.str.contains('maternal', regex= True, na=False,case=False)]
Pregnant = df[df.Title.str.contains('pregnant', regex= True, na=False,case=False)]
Cancer = df[df.Title.str.contains('cancer', regex= True, na=False,case=False)]
Pneumonia = df[df.Title.str.contains('pneumonia', regex= True, na=False,case=False)]
Asthama = df[df.Title.str.contains('asthama', regex= True, na=False,case=False)]
Diarrhea = df[df.Title.str.contains('diarrhea', regex= True, na=False,case=False)]
Hepatitis = df[df.Title.str.contains('hepatitis', regex= True, na=False,case=False)]
Rheumatic =  df[df.Title.str.contains('Rheumatic', regex= True, na=False,case=False)]
Silicosis = df[df.Title.str.contains('silicosis', regex= True, na=False,case=False)]
Mortality_Risk = df[df.Title.str.contains('mortality risk', regex= True, na=False,case=False)]
Risk = df[df.Title.str.contains('risk', regex= True, na=False,case=False)]
Diagnosis = df[df.Title.str.contains('diagnosis', regex= True, na=False,case=False)]
Prognosis = df[df.Title.str.contains('prognosis', regex= True, na=False,case=False)]
Hydroxi = df[df.Title.str.contains(' hydroxychloroquine', regex= True, na=False,case=False)]
RVD = df[df.Title.str.contains(' Remdesivir', regex= True, na=False,case=False)]
ECO = df[df.Title.str.contains('Economy', regex= True, na=False,case=False)]
Social_Distancing = df[df.Title.str.contains('Social Distancing', regex= True, na=False,case=False)]
Education = df[df.Title.str.contains('Education', regex= True, na=False,case=False)]


# In[45]:


Education.shape


# In[28]:


Pneumonia_Abs =  Pneumonia[Pneumonia.Abstract.str.contains('pneumonia', regex= True, na=False,case=False)]
Pneumonia_Abs =  Pneumonia_Abs[['Title','Abstract']]
Abstract =Pneumonia_Abs['Abstract'].str.cat(sep=',')
parser = PlaintextParser(Abstract, Tokenizer("english"))
summarizer_lsa = LsaSummarizer()
summary_2 =summarizer_lsa(parser.document,15) #Summarize the document with 5 sentences


# In[11]:


# !pip3 install beautifulsoup4
# !pip3 install google
# !pip3 install colored


# In[29]:


from termcolor import colored


# In[30]:


i =1
for sentence in summary_2:
    print(colored("[summary point {}]".format(i),"red") ,colored("{}".format(sentence),"blue"))
    i = i+1
    


# In[ ]:




