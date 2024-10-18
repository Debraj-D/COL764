# Document Reranking using Word2Vec and Glove Embeddings

The goal is to evaluate different models of pseudo-relevance feedback based query expansion and reranking, aimed at improving the precision and recall of results. We use Local and pre-trained Generic embeddings to expand the query and rerank the documents using the expanded query and present the results. Please refer to the assignment pdf for further details and clarifications about the problem.

## How to Run

Build Script (*build.sh*) - This file will be used to build the source code. Essentially an empty file for submission.
```
bash build.sh
```

### Task-0 Relevance Model for Retrieval

We have used the standard Language Modeling based retrieval model with Dirichlet smoothing.
```
python3 baseLM.py [query-file] [top-100-file] [collection-file] [output-file]
```
**Example usage:**      
```
python3 baseLM.py ./COL764-A2-2024/queries.tsv ./COL764-A2-2024/top100docs.tsv ./COL764-A2-2024/docs.tsv ./output-file.txt
```

### Task-1 Local Embeddings for Query Expansion and Reranking

We use local embeddings (i.e., locally computed word embeddings) to expand the query and then do the reranking using these embeddings
```
bash w2v-local_rerank.sh [query-file] [top-100-file] [collection-file] [output-file] [expansions-file]
```
**Example usage:**
```
bash w2v-local_rerank.sh ./COL764-A2-2024/queries.tsv ./COL764-A2-2024/top100docs.tsv ./COL764-A2-2024/docs.tsv ./output-file ./expansion-file
```

### Task-2 Generic Embeddings for Query Expansion

We can find the top-m nearest words to the original query terms from a pretrained word embedding model, and using them to find top-N expansion terms. Pretrained word2vec and GloVe embeddings are used.
```
bash w2v-gen_rerank.sh [query-file] [top-100-file] [collection-file] [w2v-embeddings-file] [output-file] [expansions-file]
```

and 

```
bash glove-gen_rerank.sh [query-file] [top-100-file] [collection-file] [glove-embeddings-file] [output-file] [expansions-file]
```

**Example usage:**
```
bash w2v-gen_rerank.sh ./COL764-A2-2024/queries.tsv ./COL764-A2-2024/top100docs.tsv ./COL764-A2-2024/docs.tsv /COL764-A2-2024/word2vec.300d.txt ./output-file ./expansion-file
```

### Running TREC Eval
```
./trec_eval-9.0.7/trec_eval -m ndcg -m ndcg_cut.5,10,50 <Query-Results-File> <Output-To-Be-Tested>
```
eg. 
```
./trec_eval-9.0.7/trec_eval -m ndcg -m ndcg_cut.5,10,50 ./COL764-A2-2024/qrels.tsv ./output-file.txt
```

### Running Custom Eval (*nDCG_score.py*)
```
python3 nDCG_score.py [ground_truth_file] [results_folder]
```

eg.
```
python 3 nDCG_score.py /COL764-A2-2024/qrels.tsv ./output-file.txt
```
-----
&nbsp;  
Other Files:

- *functional.py*: functions for processing text and calculating KL Divergence
- *LangModels.py*: Contains implementations of basic Uni-gram Language models with Dirichilet Smoothing
- *tokenizer.py*: WordPunct Tokenizer and Treebank Word Tokenizer
- *nDCG_score.py*: To score reranked documents against the ground-truth
-----
&nbsp;  
Other Folders:

- *ipynb*: contains the Python Notebook files along with folders of generated expansion and output files for comparison
- *Permutations*: Contains expansion and output files for permutations of text preprocessing parameters
- *tables*: contains .csv and .numbers files for result comparison
- *COL764-A2-2024*: Contains the queries, top 100 docs for each query, ground truth files and the generic embeddings for Word2Vec and Glove
-----
&nbsp;  
Link for Word2Vec Implementation:  
https://github.com/giuseppefutia/word2vec

Here is an article for easy understanding of the implemenation:  
https://towardsdatascience.com/a-word2vec-implementation-using-numpy-and-python-d256cf0e5f28

For more details, refer to : 2021ME10973.pdf

#### P.S. Note : 
The docs file is not provided due to its large size (*~22 GB*)
The generic embeddings files are also not provided due to their large size(*~300 MB*)