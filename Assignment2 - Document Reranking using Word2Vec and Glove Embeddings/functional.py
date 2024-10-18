import os
import re
import numpy as np
import string
import nltk
import math
nltk.download('popular')
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from tokenizer import SimpleTokenizer

contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not","can't": "can not","can't've": "cannot have",
"'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
"didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
"hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
"he'll've": "he will have","how'd": "how did","how'd'y": "how do you","how'll": "how will","i'd": "i would",
"i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have",
"isn't": "is not","it'd": "it would","it'd've": "it would have","it'll": "it will","it'll've": "it will have",
"let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not",
"mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have",
"needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
"oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
"shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will",
"she'll've": "she will have","should've": "should have","shouldn't": "should not",
"shouldn't've": "should not have","so've": "so have","that'd": "that would","that'd've": "that would have",
"there'd": "there would","there'd've": "there would have",
"they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
"they're": "they are","they've": "they have","to've": "to have","wasn't": "was not","we'd": "we would",
"we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
"weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
"what've": "what have","when've": "when have","where'd": "where did",
"where've": "where have","who'll": "who will","who'll've": "who will have","who've": "who have",
"why've": "why have","will've": "will have","won't": "will not","won't've": "will not have",
"would've": "would have","wouldn't": "would not","wouldn't've": "would not have","y'all": "you all",
"y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
"you'd": "you would","you'd've": "you would have","you'll": "you will","you'll've": "you will have",
"you're": "you are","you've": "you have"}

# Expand contractions helper function
def expand_contractions(text, contractions_dict=contractions_dict):
    contractions_re = re.compile('(%s)' % '|'.join(re.escape(key) for key in contractions_dict.keys()))
    return contractions_re.sub(lambda match: contractions_dict[match.group(0)], text)

# Pre-Process Text Parameters are boolean. TRUE if you want to remove that.
def preProcessText(text,lowercase=True,contractions=False,punctuations=True,digits=False,stemming=False,Stopwords=True):

    if lowercase:
        text = text.lower()

    if contractions:
        text = expand_contractions(text)

    if Stopwords:
        stop_words = set(stopwords.words('english'))
        text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
        
    if punctuations:
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)

    if stemming:
        stemmer = SnowballStemmer("english")
        words = nltk.tokenize.wordpunct_tokenize(text)
        stemmed_words = [stemmer.stem(word) for word in words]
        text = ' '.join(stemmed_words)

    if digits:
        digits_pattern = r"\d+(\.\d+)?"
        text = re.sub(digits_pattern, "<NUMBER>", text)    

    text = ' '.join(word for word in text.split() if word.isascii())

    return text


def KL_Divergence(document_model, relevance_model_probabilities):
    doc_prob = []
    rel_prob = []
    for token in relevance_model_probabilities.keys():
        doc_prob.append(document_model.token_probability(token))
        rel_prob.append(relevance_model_probabilities[token])
    doc_prob = np.array(doc_prob)
    rel_prob = np.array(rel_prob)
    
    return np.sum(doc_prob * np.log10(doc_prob/rel_prob))

def KL_Divergence_Reverse(document_model, query_model):
    accumulator = 0
    for word in query_model.keys():
        if query_model[word] > 0:
            accumulator += query_model[word] * math.log10(query_model[word]/document_model.token_probability(word))
    return accumulator

def parse_results_file(results_file_path):    
    results = {}    # query_id -> rank -> doc_id

    with open(results_file_path, 'r') as rf:
        for line in rf.readlines():
            if line.strip():
                query_id, ignore_col, doc_id, rank, score, run_id = line.split()
                query_id = int(query_id)
                rank = int(rank)
                if query_id not in results:
                    results[query_id] = {}
                results[query_id][rank] = doc_id
    return results