import os
import random
random.seed(8446)
import copy
import myutils

positives = []
occs = set()
data_dir = 'data/TaskA/training/'
for lang in myutils.train_langs:
    train_path = data_dir + lang + '/' + os.listdir(data_dir + lang)[0]
    for line in open (train_path):
        _, _, occ1, occ2 = line.strip().split('\t')
        positives.append(occ1 + '\t' + occ2)
        occs.add(occ1)
        occs.add(occ2)

    for neg_samples in [1]:#range(1,5):
        print(neg_samples)
        all_samples = copy.deepcopy(positives)
        for item_index in range(len(all_samples)):
            all_samples[item_index] += '\t1'
        for occ in occs:
            sampled = 0
            while sampled < neg_samples:
                occ2 = random.choice(list(occs))
                if occ1 + '\t' + occ2 in positives or occ2 + '\t' + occ1 in positives:
                    continue
                all_samples.append(occ + '\t'+ occ2 + '\t0')
                sampled += 1
        random.shuffle(all_samples)
        out_file = open('data/taska.' + lang + '.neg' + str(neg_samples), 'w')
        out_file.write('\n'.join(all_samples) + '\n')
        out_file.close()


import torch
for split in ['validation', 'test']:
    val_dir = 'data/TaskA/' + split + '/'
    for lang in os.listdir(val_dir):
        occ1_names = [line.strip().split('\t')[1] for line in open(val_dir + lang + '/queries')]
        if os.path.isfile(val_dir + lang + '/corpus_elements'):
            occ2_names = [line.strip().split('\t')[1] for line in open(val_dir + lang + '/corpus_elements')]
        else:
            occ2_names = [line.strip().split('\t')[1] for line in open(val_dir + lang + '/corpus_element')]
        annotation = torch.zeros((len(occ1_names), len(occ2_names)))
        if split == 'validation':
            for line in open(val_dir + lang + '/qrels.tsv'):
                tok = line.strip().split('\t')
                occ1_idx = int(tok[0])
                occ2_idx = int(tok[2])
                annotation[occ1_idx][occ2_idx] = 1
        
        if split == 'validation':
            outFile = open('data/taska.' + lang + '.dev', 'w')
        else:
            outFile = open('data/taska.' + lang + '.test', 'w')
        for occ1_idx, occ1 in enumerate(occ1_names):
            if occ1_idx == 0:
                continue
            for occ2_idx, occ2 in enumerate(occ2_names):
                if occ2_idx == 0:
                    continue
                outFile.write(occ1 + '\t' + occ2 + '\t' + str(int(annotation[occ1_idx][occ2_idx].item())) + '\n')
            
        outFile.close()
    

