import os
import csv
import re
import sys
from collections import defaultdict
from LangModels import *
from functional import *

csv.field_size_limit(sys.maxsize)

query_file = sys.argv[1]    # "/Users/jeet/Downloads/COL764-A2-2024/queries.tsv"
top_100_file = sys.argv[2]    # "/Users/jeet/Downloads/COL764-A2-2024/top100docs.tsv"
collection_file = sys.argv[3]    # "/Users/jeet/Downloads/COL764-A2-2024/docs.tsv"
output_file = sys.argv[4]    # "output-file"

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

# Make Docs dictionary
with open(collection_file,'r') as cf:
    tsv_reader = csv.reader(cf,delimiter='\t')
    for row in tsv_reader:
        if row[0] in DocIDs:
            Docs[row[0]] = {'url':row[1],'title':row[2],'body':row[3]}
        if len(Docs)==2400:
            break

# For each Query
query_count = 0
for qid,qtext in Queries.items():
    query_count +=1
    query = preProcessText(qtext,lowercase=True,punctuations=True,digits=False,stemming=False,Stopwords=True,contractions=True)

    # Get Language Model for each of the top 100 doc for the query containing token freq and doc length
    doc_lang_models = []
    for rank,(doc_id,score) in Top100[qid].items():
        doc_text = Docs[doc_id]['title'] + " " + Docs[doc_id]['body']
        doc_lang_models.append(DocLangModel(doc_id,doc_text,'S1'))
    
    # Get Collection Statistics - Collection is of the top 100 docs
    Coll_lang_model  = CollectionLangModel()
    for DLM in doc_lang_models:
        Coll_lang_model.add_DocLangModel(DLM)

    Coll_lang_model.add_unk(0.5)

    # Add Collection Statistics to each Document language Model to calculate M_d(t) = (f_(t,d) + mu * M_c(t))/(l_d + mu)
    for DLM in doc_lang_models:
        DLM.add_collection_model(Coll_lang_model)

    query_tokens = [token if token in Coll_lang_model.coll_token_freq.keys() else '<UNK>' for token in query.split()]
    # print(query_tokens)

    Query_prob_per_doc = []     # Probability of entering q as a query given that d is a relevant document
    for DLM in doc_lang_models:
        query_prob = 1
        for term in query_tokens:
            query_prob *= DLM.token_probability(term)
        Query_prob_per_doc.append(query_prob)  
    
    Avg_query_prob = sum(Query_prob_per_doc)/len(Query_prob_per_doc)

    # compute relevant model probabilities
    relevance_model_prob = defaultdict(int)

    for token in Coll_lang_model.coll_token_freq.keys():
        for idx,DLM in enumerate(doc_lang_models):
            prob_TGD = DLM.token_probability(token)     # Probability of the token being in the relevant document (TGD - Term Given Document)
            relevance_model_prob[token] += prob_TGD * Query_prob_per_doc[idx]
        relevance_model_prob[token] /= len(doc_lang_models)
        relevance_model_prob[token] /= Avg_query_prob

    results = []
    for i in range(len(doc_lang_models)):
        results.append((doc_lang_models[i].doc_id, 1-KL_Divergence(doc_lang_models[i], relevance_model_prob)))
    
    results.sort(key=lambda x: x[1], reverse=True)

    with open(output_file,'a') as out:
        for idx, (doc_id,score) in enumerate(results):
            out.write(f"{qid} Q0 {doc_id} {idx+1} {score:.6f} runid1\n")

    print(f"Processed Query Number: {query_count}\r")




