import csv
import nltk

# download csv file from https://www.kaggle.com/jrobischon/wikipedia-movie-plots, save as movie_plots.csv and place in same directory

# dictionary format- title: (year, plot)
movies = {}

with open('movie_plots.csv') as plots:
    csv_reader = csv.reader(plots, delimiter=',')
    for row in csv_reader:
        if row[2] == 'American':
            movies[row[1]] = (row[0], row[7])

