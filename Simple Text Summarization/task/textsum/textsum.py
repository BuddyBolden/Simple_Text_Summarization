# Write your code here
import lxml
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from bs4 import BeautifulSoup
import math

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

stop_words = stopwords.words('english')
model = TfidfVectorizer(tokenizer=word_tokenize)
xml_file = r"C:\Users\necas\PycharmProjects\Simple Text Summarization\Simple Text Summarization\task\news.xml"
with open(xml_file, "r") as f:
    content = f.read()

soup = BeautifulSoup(content, "xml")
#print(soup.prettify())
lemmatizer = WordNetLemmatizer()
news = soup.find_all("news")
for article in news:
    header = article.find("value", {"name":"head"})
    header_tokenized = word_tokenize(header.text)
    header_preprocessed = []
    for word in header_tokenized:
        if word not in string.punctuation:
            if word not in stop_words:
                edited_word = lemmatizer.lemmatize(word.lower(), pos="n")
                if edited_word not in stop_words:
                    header_preprocessed.append(edited_word)



    print(f"HEADER: {header.text}")



    news_text = article.find("value", {"name":"text"}).text
    text_list = sent_tokenize(news_text)
    word_list = []
    for sentence in text_list:
        sentence_words = []
        sentence_words = word_tokenize(sentence)
        word_list.append(sentence_words)
    word_list2 = []
    for sentence in word_list:
        sentence_words2 = []
        for word in sentence:
            if word not in string.punctuation:
                if word not in stop_words:
                    edited_word =  lemmatizer.lemmatize(word.lower(), pos="n")
                    if edited_word not in stop_words:
                        sentence_words2.append(edited_word)
        word_list2.append(sentence_words2)



    lentext = (round(math.sqrt(len(text_list))))
    preprocessed_list = [' '.join(sentence) for sentence in word_list2]
    matrix = (model.fit_transform(preprocessed_list)).toarray()
    sentence_ranking = {}

    for i, sentence in enumerate(word_list2):
        for word in sentence:
            word_index = model.vocabulary_.get(word)
            if word_index is not None:
                if word in header_preprocessed:
                    matrix[i, word_index] *= 3

        sentence_matrix = matrix[i, matrix[i, :] != 0]
        sentence_ranking[i] = np.mean(sentence_matrix) if sentence_matrix.size > 0 else 0



    sentence_ranking2 = dict(sorted(sentence_ranking.items(), key=lambda item: item[1], reverse=True))
    sentence_ranking2_list = list(sentence_ranking2.keys())[:lentext]
    sentence_ranking2_list.sort()
    print(f"TEXT: {text_list[sentence_ranking2_list[0]]}")
    for index in sentence_ranking2_list[1:]:
        print(text_list[index])

    print("")


