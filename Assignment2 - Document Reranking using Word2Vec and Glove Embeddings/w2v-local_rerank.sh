#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 5 ]; then
    echo "Usage: w2v-local_rerank.sh [query-file] [top-100-file] [collection-file] [output-file] [expansions-file]"
    exit 1
fi

# Assign command line arguments to variables
query_file=$1
top_100_file=$2
collection_file=$3
output_file=$4
expansions_file=$5

# Run the Python tokenizer program
python3 w2v-local_rerank.py "$query_file" "$top_100_file" "$collection_file" "$output_file" "$expansions_file"
