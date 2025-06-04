import os
import myutils

for pred_file in os.listdir('predsB'):
    if pred_file.endswith('eval') and len(open('predsB/' + pred_file).readlines()) != 14:
        out_path = 'predsB/' + pred_file.replace('.eval', '')
        conv_cmd = 'python3 scripts/convertB.py ' + out_path
        print(conv_cmd)
        os.system(conv_cmd)

        gold_path = 'data/TaskB/validation/qrels.tsv'
        eval_cmd = 'python3 talentclef25_evaluation_script/talentclef_evaluate.py --qrels ' + gold_path + ' --run ' + out_path + '.tsv'
        eval_cmd += ' > ' + out_path + '.eval'

        print(eval_cmd)
        os.system(eval_cmd)


