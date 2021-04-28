from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
nltk.download('punkt')


text_pt = "Quem poderia imaginar que programas de computador seriam capazes de analisar sentimentos no texto"
text_en = "Who would have thought that computer programs would be analyzing human sentiments"

tokens = word_tokenize(text_en)

print(tokens)


"""
    Stop words: are mostly connector words that do not contribute much to the meaning of the sentence
    usar: nltk.download('stopwords')
"""
nltk.download('stopwords')
stopwords_en = nltk.corpus.stopwords.words('english')
print("Stop words EN")
print(stopwords_en)

stopwords_pt_br = nltk.corpus.stopwords.words('portuguese')
print("Stop words pt-br")
print(stopwords_pt_br)

# Separando palavras que não são stop words
networks = [word for word in tokens if word not in stopwords_en]
print(networks)

"""Lemmanization and stemming: tecniques used to reduce words to their root form
pre-req: ltk.download('wordnet')
"""
nltk.download('wordnet')
# Uso de Lemmatization
lemmatizer = WordNetLemmatizer()
tokens_lem = [lemmatizer.lemmatize(word) for word in tokens]
print("Lemmatizer")
print(text_en)
print(tokens_lem)

# Uso de Stemming
tks = word_tokenize(text_en.lower())
ps = PorterStemmer()
token_ps = [ps.stem(word) for word in tks]
print("Stemmer")
print(token_ps)


"""POS tagging: identifies parts of the speech (nun, verb, adverb, etc)
pre-req: nltk.download('averaged_perceptron_tagger')
"""
nltk.download('averaged_perceptron_tagger')
#text2 = "Usain Bolt is the fastest runner in the word"
text2 = "Usain Bolt é o corredor mais rápido do mundo"
tokens = word_tokenize(text2)
print("POS tagging")
print([nltk.pos_tag([token]) for token in tokens])

# Lista exaustiva de NLTK POS tag
nltk.download('tagsets')
# nltk.help.upenn_tagset()

"""Textblob: Sentimental Analysis
"""

eval = TextBlob("I love pizza").sentiment
print(eval)


"""Machine Translator
"""
print("Machine Translator")
languages = ['en', 'fr', 'es']
for language in languages:
    print(TextBlob("Quem disse que tradução não é divertido?").translate(to=language))


"""POS Tagging com TextBlob
"""
tags = TextBlob("The global economy is expected to grow this year").tags
print(tags)

"""VADER: lexicon-based sentimental analyzis tool, including colloquial language terms, emoticons, slang, acronysm, etc
"""

analyzer = SentimentIntensityAnalyzer()
retorno = analyzer.polarity_scores("OMG ! This book is so cool")
print(retorno)
