import json
import pandas as pd
from textblob import Word
from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
from process_string import preprocess
from textblob import TextBlob
import os

df = pd.read_json('items.json', orient='values', typ='frame',encoding=str)
df.drop(df.columns[[0,2,3]], axis=1, inplace=True)
df['Category'] = 'Category'

train = [
    ('Laboratory products Tests Instruments Analysis data Mechanical Tests Laboratory', 'Laboratory Products'),
    ('catheterization monitoring devices operations surgery', 'Medical Equipment'),
    ('Medical equipment aid diagnosis, monitoring treatment medical conditions', 'Medical Equipment'),
    ('clothes blood injections patients are sterile', 'Disposables & Consumables'),
    ('Pediatrics nutrition aids children growth health', 'Cosmetics and nutrition'),
    ('Antibiotic capsules are manufacture chemicals pharmaceutical companies like Ranbaxy', 'Pharmaceutical Products'),
    ('Drug synonymous tablets fall under medicine pharma', 'Pharmaceutical Products')
]

"""temp = []
for sentence, sentence1 in zip(df['Description'], df['Tender_Category']):
	en_blob = TextBlob(preprocess(sentence.encode('ascii','ignore')) + " " + preprocess(sentence1.encode('ascii','ignore')))
	en_blob.translate(to='es')
	temp.append(en_blob.words)
flattened  = [val for sublist in temp for val in sublist]

temp = []
for w in flattened:
	temp.append(Word(w).lemmatize())
categories_set = set(temp)
print(categories_set)"""

cl = NaiveBayesClassifier(train)

for i, sentence in zip(range(0, len(df)-1), df['Description']):
	temp_sentence = preprocess(sentence.encode('ascii','ignore'))
	df['Category'][i] = cl.classify(temp_sentence)
	 

os.remove("output.csv")
df.to_csv('output.csv', sep=',', encoding='utf-8')

print(df[['Description', 'Category']])