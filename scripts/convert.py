import sys
import csv
import os

lang = sys.argv[1].split('/')[-1].split('.')[-2]
if 'test' in sys.argv[1]:
    split = 'test'
else:
    split = 'validation'
title_lookup = {}
for line in open('data/TaskA/' + split + '/' + lang + '/queries').readlines()[1:]:
    q_id, title = line.strip().split('\t')
    title_lookup[title] = q_id

doc_lookup = {}
if os.path.isfile('data/TaskA/' + split + '/' + lang + '/corpus_elements'):
    for line in open('data/TaskA/' + split + '/' + lang + '/corpus_elements').readlines()[1:]:
        c_id, title = line.strip().split('\t')
        doc_lookup[title] = c_id
else:
    for line in open('data/TaskA/' + split + '/' + lang + '/corpus_element').readlines()[1:]:
        c_id, title = line.strip().split('\t')
        doc_lookup[title] = c_id

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
    
