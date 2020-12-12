import gensim
from gensim.models import Word2Vec
from gensim.summarization import keywords
from nltk.corpus import brown
import csv
import numpy as np
import random
from statistics import mean

#arrays to store keywords by year and historical events
key0 = []
key10 = []
key20 = []
key30 = []
key40 = []
key50 = []
key60 = []
key70 = []
key80 = []
key90 = []
key00 = []
key21 = []
war = []
depression = []
postwar = []

#read in keywords csv
with open('year_words.csv') as plots:
    csv_reader = csv.reader(plots, delimiter=',')
    for row in csv_reader:
            # if int(row[0]) < 1910:
            #     key0.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 1920:
            #     key10.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 1930:
            #     key20.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 1940:
            #     key30.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 1950:
            #     cross.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 1960:
            #     key21.append(tuple((row[0], row[1])))
            #     key50.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 1970:
            #     key60.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 1980:
            #     key70.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 1990:
            #     key80.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 2000:
            #     key90.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 2010:
            #     key00.append(tuple((row[0], row[1])))
            # elif int(row[0]) < 2020:
            #     cross.append(tuple((row[0], row[1])))
            if int(row[0]) >= 1914 and int(row[0]) <= 1914:
                war.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 1915 and int(row[0]) <= 1920:
                postwar.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 1929 and int(row[0]) <= 1933:
                depression.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 1939 and int(row[0]) <= 1945:
                war.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 1950 and int(row[0]) <= 1953:
                war.append(tuple((row[0], row[1])))
                wp.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 1954 and int(row[0]) <= 1959:
                postwar.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 1946 and int(row[0]) <= 1949:
                postwar.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 2006 and int(row[0]) <= 2009:
                depression.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 1975 and int(row[0]) <= 1980:
                postwar.append(tuple((row[0], row[1])))
            elif int(row[0]) >= 1990 and int(row[0]) <= 1995:
                postwar.append(tuple((row[0], row[1])))

#brown corpus model
model = gensim.models.Word2Vec(brown.words())

#select random word from array
rand = postwar.pop(random.randrange(len(postwar)))
rand_word = rand[1]
res = [key[1] for key in postwar]
values = []
#loop through 50 possible keywords to find average similarity
for i in range(50):
    r = res.pop(random.randrange(len(res)))
    if r in brown.words():
        cos_sim = model.wv.n_similarity(rand_word, r)
        values.append(cos_sim)
print("mean: " + str(mean(values)))
