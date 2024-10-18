import csv
import os
from math import log2
from functional import parse_results_file

ground_truth_file = sys.argv[1]    # "/Users/jeet/Downloads/COL764-A2-2024/qrels.tsv"
results_folder = sys.argv[2]    # "/Users/jeet/Desktop/IIT Delhi/7th Sem/COL764/Assignments/Assignment2"
output_file = "final_answer.csv"
ground_truth = {}   # query_id : { doc_id : relevance }

# Reading the ground truth file
with open(ground_truth_file,'r') as gt:
    tsv_reader = csv.reader(gt,delimiter='\t')
    next(tsv_reader)
    query_id = -1
    for row in tsv_reader:
        if int(row[0]) != query_id:
            query_id = int(row[0])
            ground_truth[query_id] = {}
        ground_truth[query_id][row[1]] = int(row[2])

# Open the output file for writing results
with open(output_file, 'w') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["Filename", "Avg_nDCG@5", "Avg_nDCG@10", "Avg_nDCG@50"])

    # Loop through files in the results_folder that start with 'output'
    for file_name in os.listdir(results_folder):
        if file_name.startswith("output"):
            results_file = os.path.join(results_folder, file_name)

            output_results = parse_results_file(results_file)

            query_ids = list(map(int, output_results.keys()))

            avg_nDCG_5 = 0
            avg_nDCG_10 = 0
            avg_nDCG_50 = 0

            for qid in query_ids:
                gt_scores = [(score, doc_id) for doc_id, score in ground_truth[qid].items()]
                sorted_gt_scores = sorted(gt_scores, key=lambda x: x[0], reverse=True)

                output_scores = {}      # doc_rank : relevance
                for rank, doc_id in output_results[qid].items():
                    if ground_truth[qid].get(doc_id) != None:
                        output_scores[rank] = ground_truth[qid][doc_id]  
                    else :
                        output_scores[rank] = 0
                
                max_rank = max(output_scores.keys())       # 100
                gains = [output_scores[rank] for rank in range(1,max_rank+1)]       # relevance at ranks

                # Calculate DCG
                discounted_gains = [score/log2(idx+2) for idx, score in enumerate(gains)]

                DCG_at_ranks, DCG = [], 0
                for dg in discounted_gains:
                    DCG += dg
                    DCG_at_ranks.append(DCG)

                # Calculate IDCG (ideal DCG)
                ideal_gains = []
                for rank in range(0,max_rank):
                    if rank < len(sorted_gt_scores):
                        ideal_gains.append(sorted_gt_scores[rank][0])
                    else:
                        ideal_gains.append(0)

                ideal_discounted_gains = [score/log2(idx+2) for idx, score in enumerate(ideal_gains)]

                IDCG_at_ranks, IDCG = [], 0
                for idg in ideal_discounted_gains:
                    IDCG += idg
                    IDCG_at_ranks.append(IDCG)

                def normalized_score_at_rank(rank):
                    return DCG_at_ranks[rank-1]/IDCG_at_ranks[rank-1]
                
                avg_nDCG_5 += normalized_score_at_rank(5) 
                avg_nDCG_10 += normalized_score_at_rank(10) 
                avg_nDCG_50 += normalized_score_at_rank(50)
                # print(f"Query: {qid}\n\tnDCG@5: {normalized_score_at_rank(5)}\n\tnDCG@10: {normalized_score_at_rank(10)}\n\tnDCG@50: {normalized_score_at_rank(50)}")
            
            # Calculate average nDCG values and write to file
            avg_nDCG_5 /= len(query_ids)
            avg_nDCG_10 /= len(query_ids)
            avg_nDCG_50 /= len(query_ids)

            writer.writerow([file_name, avg_nDCG_5, avg_nDCG_10, avg_nDCG_50])
print('done')
# print(f"Average nDCG@5 : {avg_nDCG_5/(len(query_ids))}")
# print(f"Average nDCG@10 : {avg_nDCG_10/(len(query_ids))}")
# print(f"Average nDCG@50 : {avg_nDCG_50/(len(query_ids))}")