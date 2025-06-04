import os
import random
random.seed(8446)
import copy
import myutils
import ast

import pandas as pd


for split in ['validation', 'test']:
    val_dir = 'data/TaskB/' + split + '/'
    corpus_path = val_dir + 'corpus_elements'
    corpus_df = pd.read_csv(corpus_path, sep="\t", names=["c_id", "esco_uri", "skill_aliases"], skiprows=1)
    
    corpus_df["skill_aliases"] = corpus_df["skill_aliases"].apply(lambda x: ast.literal_eval(x))
    corpus_df["skill_text"] = corpus_df["skill_aliases"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    
    skills = {}
    for titles, c_id in zip(corpus_df['skill_aliases'], corpus_df['c_id']):
        skills[c_id] = titles[0]
    
    jobtitles = {}
    for line in open(val_dir + 'queries').readlines()[1:]:
        tok = line.strip().split('\t')
        jobtitles[tok[0]] = tok[1] # Alternative random/all
    
    
    #skills = {}
    #for line in open(val_dir + 'corpus_elements').readlines()[1:]:
    #    tok = line.strip().split('\t')
    #    skills_list = ast.literal_eval(tok[-1])
    #    if skills_list[0] == '[':
    #        print(tok[-1][1:-1])
    #        print(ast.literal_eval(tok[-1][1:-1]))
    #    skills[tok[0]] = ast.literal_eval(tok[-1])[0]
    
    positives = {job: [] for job in jobtitles}
    
    if split != 'test':
        for line in open(val_dir + 'qrels.tsv'):
        
            tok = line.strip().split('\t')
            occ_id = tok[0]
            skill_id = tok[2]
            positives[occ_id].append(skill_id)
    
    if split == 'validation':
        outFile = open('data/taskb.dev', 'w')
    else:
        outFile = open('data/taskb.test', 'w')
    for job in sorted(jobtitles):
        job_name = jobtitles[job]
        for skill in skills:
            if skill in positives[job]:
                outFile.write(job_name + '\t' + skills[skill] + '\t1\n')
            else:
                outFile.write(job_name + '\t' + skills[skill] + '\t0\n')
    outFile.close()
        

import json
train_dir = 'data/TaskB/training/'
job_data = json.load(open(train_dir + 'jobid2terms.json'))
skill_data = json.load(open(train_dir + 'skillid2terms.json'))

all_positives = {job: [] for job in job_data}
for line in open(train_dir + 'job2skill.tsv'):
    tok = line.strip().split('\t')
    occ_id, skill_id, _ = tok
    all_positives[occ_id].append(skill_id)

import random
random.seed(8446)
for neg in [1,2,5]:
    out_file = open('data/taskb.train.neg' + str(neg), 'w')
    for job in all_positives:
        all_negs = list(skill_data.keys())
        for positive in all_positives[job]:
            if positive in all_negs:
                all_negs.remove(positive)
            
            out_file.write(job_data[job][0] + '\t' + skill_data[positive][0] + '\t1\n')
        random.shuffle(all_negs)
        for i in range(len(all_positives[job]) * neg):
            out_file.write(job_data[job][0] + '\t' + skill_data[all_negs[i]][0] + '\t0\n')
    
    out_file.close()

