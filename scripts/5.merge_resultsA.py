import os

import myutils
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def getMAP(path):
    for line in open(path):
        if line.startswith('map'):
            return float(line.strip().split(' ')[-1])
    return 0.0
table_data = {}
for lang in myutils.train_langs + ['chinese']:
    models = {}
    for eval_file in sorted(os.listdir('preds')):
        if not eval_file.endswith('eval'):
            continue
        if lang not in eval_file:
            continue
        tok = eval_file.split('.')
        model = '.'.join(tok[:-2])
        epoch = int(tok[-2])
        if model not in models:
            models[model] = [0.0] * 3
        models[model][epoch-1] = getMAP('preds/' + eval_file)
    
    plt.style.use('scripts/rob.mplstyle')
    fig, ax = plt.subplots(figsize=(8,5), dpi=300)
    print(len(models))
    
    for model in models:
        name = model.replace('multilingual.', '')
        name = name[:name.find('1-lessepochs3')-1]
        if name not in table_data:
            table_data[name] = {}
        print(model, models[model])
        table_data[name][lang] = max(models[model])
        ax.plot(range(1,4), models[model], label=model[model.find('.')+1:])

    leg = ax.legend(loc='lower right')
    leg.get_frame().set_linewidth(1.5)
    
    ax.set_ylabel('MAP')
    ax.set_xlabel('epoch')
    fig.savefig('all_map-' + lang + '.pdf', bbox_inches='tight')


for lm in table_data:
    scores = [lm]
    total = 0.0
    for lang in ['english', 'german', 'spanish', 'chinese']:
        if lang in table_data[lm]:
            scores.append('{:.4f}'.format(table_data[lm][lang]))
            total += table_data[lm][lang]
        else:
            scores.append('0.0000')
    scores.append('{:.4f}'.format(total/4))
    print(' & '.join(scores) + ' \\\\')

