import csv
import nltk
import string
import random
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier


stop_words = stopwords.words('english')
# dictionary format- title: (year, plot)
pos_movies = {}
# dictionary format- title: (year, plot)
neg_movies = {}
# dictionary format- title: [tokens]
m_tokens = {}
# dictionary format- title: [tokens]
pm_tokens = {}
# dictionary format- title: [tokens]
nm_tokens = {}

# dictionary format- title: (year, plot)
movies = {}
# dictionary format- title: (year, classification)
classified_movies = {}

# dictionary format- year: num of positives
pos_counts = {}
# dictionary format- year: num of negatives
neg_counts = {}

# used to classify some plots for training and testing purposes
def hand_classify(genre):
    if 'war' in genre or 'crime' in genre or 'horror' in genre or 'melodrama' in genre or 'propaganda' in genre:
        return 'negative'
    elif 'comedy' in genre or 'animation' in genre or 'romance' in genre or 'family' in genre:
        return 'positive'

# parses, lemmatizes and filters plot text 
def process(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = []

    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        token = lemmatizer.lemmatize(word, pos)
        
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# returns all tokens in the dictionary in a list
def all_tokens(dic):
    toks = []
    for tokens in dic:
        for token in tokens:
            toks.append(token)
    return toks

# returns tokens in the format needed for the model
def tokens_model(dic):
    tok_dics = []
    for title, tokens in dic.items():
        tok_dics.append(dict([token, True] for token in tokens))
    return tok_dics

# opens file and uses process to clean text and store it in appropriate dictionaries
with open('movie_plots.csv') as plots:
    csv_reader = csv.reader(plots, delimiter=',')
    c = 1
    for row in csv_reader:
        tokens = process(row[7])
        movies[row[1]] = (row[0], row[7])
        m_tokens[row[1]] = tokens
        # fills out dictionary with training and testing data
        if hand_classify(row[5]) == 'negative':
            neg_movies[row[1]] = (row[0], row[7])
            nm_tokens[row[1]] = tokens
        elif hand_classify(row[5]) == 'positive':
            pos_movies[row[1]] = (row[0], row[7])
            pm_tokens[row[1]] = tokens
        print('row '+ str(c) +' processed')
        c += 1

all_pos_words = all_tokens(pm_tokens)
all_neg_words = all_tokens(nm_tokens)
freq_dist_pos = FreqDist(all_pos_words)
freq_dist_neg = FreqDist(all_neg_words)

p_model_toks = tokens_model(pm_tokens)
n_model_toks = tokens_model(nm_tokens)

# builds dictionaries for hand labeled postive and negative testing/training data
p_data = [(tdict, "Positive") for tdict in p_model_toks]
n_data = [(tdict, "Negative") for tdict in n_model_toks]
# joins and shuffles all test/training data 
full_data = p_data + n_data
random.shuffle(full_data)

# divides training/testing data on a 70-30 split
train_data = full_data[:5000]
test_data = full_data[5000:]

# trains and tests model
classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))

# uses trained model to classify all movies
for title, tokens in m_tokens.items():
    year = movies[title][0]
    classified_movies[title] = (year, classifier.classify(dict([token, True] for token in tokens)))

# prefills count dictionaries with all the years as keys
for year in range(1901, 2018):
    pos_counts[str(year)] = 0
    neg_counts[str(year)] = 0
pos_counts['Release Year'] = 0
neg_counts['Release Year'] = 0

# finds and stores number of positive and negative movies for each year
for value in classified_movies.values():
    if value[1] == "Positive":
        pos_counts[value[0]] = pos_counts[value[0]] + 1
    elif value[1] == "Negative":
        neg_counts[value[0]] = neg_counts[value[0]] + 1

print("Positive Counts Per Year:")
print(pos_counts)
print("Negative Counts Per Year:")
print(neg_counts)
