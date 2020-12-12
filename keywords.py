import csv
import nltk
from nltk.corpus import names
import gensim
from gensim.models import Word2Vec
from gensim.summarization import keywords
# download csv file from https://www.kaggle.com/jrobischon/wikipedia-movie-plots, save as movie_plots.csv and place in same directory

# dictionary format- title: (year, plot)
movies = {}

#read in data by year, plot, title
with open('data.csv') as plots:
    csv_reader = csv.reader(plots, delimiter=',')
    for row in csv_reader:
            movies[row[1]] = (row[0], row[7], row[1])

wy = [] # array of tuples to store year and keyword for a plot
#all common names in names corpus
namelist = ([name.lower() for name in names.words('male.txt')] +
                [name.lower() for name in names.words('female.txt')])
m = [] # tracks movie titles added to list

for movie in movies.values():
    if len(movie) > 1:
        year = movie[0]
        plot = movie[1]
        mv = movie[2]
        res = keywords(plot, words=10, split=True,lemmatize=True) # function to find keyword
        pos_res = nltk.pos_tag(res) # part of speech tagging
        #returns first (highest) keyword for a plot
        for i in pos_res:
            if i[0] not in namelist and i[1] == 'NN': #filters out common names and returns only nouns
                wy.append(tuple((year, i[0]))) # adds year, keyword to wy array
                m.append(mv) # adds title to m array
                break;

#stores results to a csv file
#two versions of this were run- one with titles and one without (year_words.csv)
with open('year_movie_words.csv', mode='w') as year_words:
    words_writer = csv.writer(year_words, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    j = 0
    for i in wy:
        words_writer.writerow([i[0], m[j], i[1]])
        j = j+1
