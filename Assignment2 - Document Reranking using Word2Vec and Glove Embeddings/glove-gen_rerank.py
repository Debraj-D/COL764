import os
import csv
import re
import sys
from collections import defaultdict,Counter
from LangModels import *
from functional import *

csv.field_size_limit(sys.maxsize)

query_file = sys.argv[1]    # "/Users/jeet/Downloads/COL764-A2-2024/queries.tsv"
top_100_file = sys.argv[2]    # "/Users/jeet/Downloads/COL764-A2-2024/top100docs.tsv"
collection_file = sys.argv[3]    # "/Users/jeet/Downloads/COL764-A2-2024/docs.tsv"
glove_embeddings_file = sys.argv[4]    # "/Users/jeet/Downloads/COL764-A2-2024/glove.6B.300d.txt"
output_file = sys.argv[5]    # "output-file"
expansion_file = sys.argv[6]    # "expansion-file"

Queries = {}    # query_id -> text
Top100 = {}     # query_id -> { rank -> (doc_id,score) }
DocIDs = set()  # set of doc_id
Docs = {}       # doc_id -> { url,title,body }

# Make Queries dictionary
with open(query_file,'r') as qf:
    tsv_reader = csv.reader(qf,delimiter='\t')
    next(tsv_reader)
    for row in tsv_reader:
        Queries[row[0]] = row[1]

print("Queries Loaded")

# Make Top100 dictionary & DocIDs set
with open(top_100_file,'r') as top:
    tsv_reader = csv.reader(top,delimiter='\t')
    next(tsv_reader)
    query = ''
    rank = 1
    for row in tsv_reader:
        if row[0]!= query:
            query = row[0]
            rank = 1
            Top100[row[0]] = {}
        Top100[row[0]][rank] = (row[1],row[2])
        DocIDs.add(row[1])
        rank+=1

print("Top 100 Doc IDs Loaded for all queries")

# Make Docs dictionary
print("Loading Doc Texts...")
with open(collection_file,'r') as cf:
    tsv_reader = csv.reader(cf,delimiter='\t')
    for row in tsv_reader:
        if row[0] in DocIDs:
            Docs[row[0]] = {'url':row[1],'title':row[2],'body':row[3]}
        if len(Docs)==2400:
            break
print("Top 100 Doc Text Loaded for all queries")

# Make Embeddings array and dictionaries to map Embedding vector indexes and the respective words
word_to_index = {}
embeddings = []
index_to_word = {}

with open(glove_embeddings_file, 'r') as glove:
    idx = 0
    for line in glove:
        line = line.strip().split()
        word = line[0].lower()
        embedding = np.array(list(map(float, line[1:])))
        word_to_index[word] = idx
        index_to_word[idx] = word
        embeddings.append(embedding/np.sqrt(np.sum(embedding**2)))   # normalize embeddings
        idx += 1
print("Word Vectors Loaded")

embeddings = np.array(embeddings)    
V, k = embeddings.shape     # embeddings is |V| * |k| where k is the dimension of the word embeddings, |V| is vocabulary size
Top_N = 20      # Number of Query Expansion Terms
Model_Lambda = 0.15      # Weight given to Expanded Query Terms

with open(expansion_file,'a') as expansion:
    query_count = 0
    for qid,qtext in Queries.items():   # query_id, query_text
        query_count +=1
        expansion.write(f"{qid}: ")
        query = preProcessText(qtext,lowercase=True,Stopwords=True,contractions=False,punctuations=True,digits=True,stemming=False)

        original_QT = SimpleTokenizer2().tokenize(query)     # Original Query Terms
        
        query_vector = np.zeros(shape=(V,1))    # binary vector to represent terms present in query
        for word, idx in word_to_index.items():
            query_vector[idx, 0] = original_QT.count(word.lower())
        
        temp = np.matmul(embeddings.T,query_vector)
        query_sim_scores = np.matmul(embeddings, temp)      # Similarity scores of trained W2V words to the query terms

        flat_sim_matrix = query_sim_scores.flatten()
        sorted_idx = np.argsort(flat_sim_matrix)[::-1]      # Sort the similaity scores in descending order and store the indexes

        sim_word_idx = [(idx, 0) for idx in sorted_idx[:Top_N]]     # Select the top N words

        expanded_QT = []    # Expanded Query Terms
        
        # Write expansion terms to Expansions file
        for term_idx,_ in sim_word_idx:
            expansion.write(f"{index_to_word[term_idx]}, ")
            score = flat_sim_matrix[term_idx]
            expanded_QT.append((index_to_word[term_idx], score))
        expansion.write("\n")

        norm_c = sum([score for word, score in expanded_QT])
        EQT_score = {k: v/norm_c for k, v in expanded_QT}   # Normalised scores for expanded query term.    Expansion Term : score
        OQT_score = Counter(original_QT)                    # Original Query Term : 1

        # Get Language Model for each of the top 100 doc for the query containing token freq and doc length
        doc_lang_models = []
        for rank,(doc_id,score) in Top100[qid].items():
            doc_text = Docs[doc_id]['title'] + " " + Docs[doc_id]['body']
            doc_lang_models.append(DocLangModel(doc_id,doc_text,'S2'))
        
        # Get Collection Statistics - Collection is of the top 100 docs
        Coll_lang_model  = CollectionLangModel()
        for DLM in doc_lang_models:
            Coll_lang_model.add_DocLangModel(DLM)

        Coll_lang_model.add_unk(0.5)

        # Add Collection Statistics to each Document language Model to calculate M_d(t) = (f_(t,d) + mu * M_c(t))/(l_d + mu)
        for DLM in doc_lang_models:
            DLM.add_collection_model(Coll_lang_model)

        # Recalculate Original Query Term scores only for terms present in the collection else <UNK>
        new_OQT_score = defaultdict(int)
        for QT, score in OQT_score.items():
            key = QT if Coll_lang_model.coll_token_freq.get(QT) is not None else '<UNK>'
            new_OQT_score[key] += score
        OQT_score = new_OQT_score

        # Recalculate Expanded Query Term scores only for terms present in the collection else <UNK>
        new_EQT_score = defaultdict(int)
        for QT, score in EQT_score.items():
            key = QT if Coll_lang_model.coll_token_freq.get(QT) is not None else '<UNK>'
            new_EQT_score[key] += score
        EQT_score = new_EQT_score

        EQT_norm = sum(EQT_score.values())  # EQT scores normalization constant
        OQT_norm = sum(OQT_score.values())  # OQT scores normalization constant

        # compute relevance model probabilities
        relevance_model_prob = defaultdict(int)
        for token in Coll_lang_model.coll_token_freq.keys():
            relevance_model_prob[token] += (Model_Lambda) * (EQT_score.get(token,0)/EQT_norm)
            relevance_model_prob[token] += (1-Model_Lambda) * (OQT_score.get(token,0)/OQT_norm)

        results = []
        for i in range(len(doc_lang_models)):
            results.append((doc_lang_models[i].doc_id, 1-KL_Divergence_Reverse(doc_lang_models[i], relevance_model_prob)))

        results.sort(key=lambda x: x[1], reverse=True)

        with open(output_file,'a') as out:
            for idx, (doc_id,score) in enumerate(results):
                out.write(f"{qid} Q0 {doc_id} {idx+1} {score:.6f} runid1\n")
        
        print(f"Processed Query Number: {query_count}\r")