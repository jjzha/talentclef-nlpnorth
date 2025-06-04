import os
import myutils



#for language in myutils.train_langs:
#    for neg_sampling in ['1', '3']:
#        for lm in myutils.llms:
#            for i  in range(1,21):
#                lm = lm.split('/')[-1]
#                if lm == 'xlm-roberta-large':
#                    continue
#                name = 'multilingual.' + lm + '.'  + neg_sampling
#                out_path = 'preds/' + name + '.' + language + '.' + str(i)
#                if not os.path.isfile(out_path):
#                    continue
#                conv_cmd = 'python3 scripts/convert.py ' + out_path
#                print(conv_cmd)
#                os.system(conv_cmd)
#
#                gold_path = 'data/TaskA/validation/' + language + '/qrels.tsv'
#                eval_cmd = 'python3 talentclef25_evaluation_script/talentclef_evaluate.py --qrels ' + gold_path + ' --run ' + out_path + '.tsv'
#                eval_cmd += ' > ' + out_path + '.eval'
#
#                print(eval_cmd)
#                os.system(eval_cmd)


for pred_file in os.listdir('preds'):
    if pred_file.endswith('eval') and len(open('preds/' + pred_file).readlines()) != 14 and 'test' not in pred_file:
        out_path = 'preds/' + pred_file.replace('.eval', '')
        conv_cmd = 'python3 scripts/convert.py ' + out_path
        print(conv_cmd)
        os.system(conv_cmd)
        language = out_path.split('.')[3]

        gold_path = 'data/TaskA/validation/' + language + '/qrels.tsv'
        eval_cmd = 'python3 talentclef25_evaluation_script/talentclef_evaluate.py --qrels ' + gold_path + ' --run ' + out_path + '.tsv'
        eval_cmd += ' > ' + out_path + '.eval'

        print(eval_cmd)
        os.system(eval_cmd)
        #exit(1)

