from collections import defaultdict
from tokenizer import *
from functional import *

DIRICHLET_MU = 750

class DocLangModel:     # Document Maximnum Likelihood Language Model with Dirichlet Smoothing
    def __init__(self,doc_id,doc_text,tokenizer='S1'):
        self.doc_id = doc_id
        self.doc_text = preProcessText(doc_text,lowercase=True,Stopwords=True,contractions=True,punctuations=True,digits=True,stemming=False)
        self.token_freq = defaultdict(int)
        if tokenizer == "S1":
            self.tokenizer = SimpleTokenizer()
        else:
            self.tokenizer = SimpleTokenizer2()
        tokens = self.tokenizer.tokenize(self.doc_text)
        self.doc_length = len(tokens)
        for token in tokens:
            self.token_freq[token] += 1
        self.collection_model = None
        self.dirichlet_mu = DIRICHLET_MU

    def add_collection_model(self,CollectionLangModel):
        self.collection_model = CollectionLangModel
    
    def token_probability(self,token):
        return ((self.token_freq.get(token,0) + self.dirichlet_mu * self.collection_model.calc_M_c(token))/(self.doc_length + self.dirichlet_mu))


class CollectionLangModel:     # Maximnum Likelihood Language Model based on term frequencies in the collection as a whole
    def __init__(self):
        self.coll_total_tokens = 0
        self.coll_token_freq = defaultdict(int)

    def add_DocLangModel(self,docModel:DocLangModel):
        self.coll_total_tokens += docModel.doc_length
        for token,freq in docModel.token_freq.items():
            self.coll_token_freq[token] += freq

    def add_unk(self, percentage):
        self.coll_token_freq['<UNK>'] = int(percentage/100 * self.coll_total_tokens)
    
    def calc_M_c(self,token):
        return (self.coll_token_freq.get(token,0)/self.coll_total_tokens)
