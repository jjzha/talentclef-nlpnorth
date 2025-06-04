import sys
import csv
import pandas as pd
import ast

if 'test' in sys.argv[1]:
    split = 'test'
else:
    split = 'validation'
title_lookup = {}
for line in open('data/TaskB/' + split + '/queries').readlines()[1:]:
    q_id, title = line.strip().split('\t')
    title_lookup[title] = q_id

corpus_path = 'data/TaskB/' + split + '/corpus_elements'
corpus_df = pd.read_csv(corpus_path, sep="\t", names=["c_id", "esco_uri", "skill_aliases"], skiprows=1)

corpus_df["skill_aliases"] = corpus_df["skill_aliases"].apply(lambda x: ast.literal_eval(x))
corpus_df["skill_text"] = corpus_df["skill_aliases"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")

doc_lookup = {}
for titles, c_id in zip(corpus_df['skill_aliases'], corpus_df['c_id']):
    doc_lookup[titles[0]] = c_id

#run_df.columns = ["q_id", "Q0", "doc_id", "rank", "score", "tag"]

with open(sys.argv[1] + '.tsv', 'w') as file:
    writer = csv.writer(file, delimiter='\t')
    #writer.writerow(["q_id", "Q0", "doc_id", "rank", "score", "tag"])

    data = {}
    for line in open(sys.argv[1]):
        src, tgt, score = line.strip().split('\t')
        if True:#if score.startswith('1') or score.startswith('0=0.5') or score.startswith('0=0.6'):
            if score.startswith('0'):
                score = 1-float(score.split('|')[0][2:])
            else:
                score = float(score.split('|')[0][2:])
        #if score.startswith('1'):# or score.startswith('0=0.5') or score.startswith('0=0.6'):
        #    if score.startswith('0'):
        #        score = 1-float(score.split('|')[0][2:])
        #    else:
        #    score = float(score.split('|')[0][2:])

            if src not in data:
                 data[src] = []
            data[src].append((score, tgt))
    print(len(data))

    for src in data:
        q_id = title_lookup[src]
        q_zero = 'Q0'
        tag = 'NLPnorth'
        for tgt_idx, tgt in enumerate(sorted(data[src], reverse=True)):
            #if tgt_idx > 1:
            #    continue
            
            rank = tgt_idx + 1
            score = tgt[0]/2
            doc_id = doc_lookup[tgt[1]]
            writer.writerow([q_id, q_zero, doc_id, rank, score, tag])
    
