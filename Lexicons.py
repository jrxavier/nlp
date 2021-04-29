# Snowball suporta várias linguas para função de Stemmer
from nltk.stem.snowball import SnowballStemmer


class Stemming(object):

    def __init__(self):
        self.plurals = ['caresses', 'flies', 'dies', 'mules', 'died']

    def getLanguagesSnowBall(self):
        print(SnowballStemmer.languages)

    def getSingles(self):
        stemmer = SnowballStemmer(language='english')
        singles = [stemmer.stem(plural) for plural in self.plurals]
        print(' '.join(singles))
