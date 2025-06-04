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
models = {}
for eval_file in sorted(os.listdir('predsB')):
    if not eval_file.endswith('eval'):
        continue
    tok = eval_file.split('.')
    model = '.'.join(tok[:-2]).replace('-lessepochs3', '')
    print(model)
    epoch = int(tok[-2])
    negs = int(model[-1])
    model = model[:-1]
    print(model)

    if model not in models:
        models[model] = [0.0] * 6
    models[model][negs] = getMAP('predsB/' + eval_file)
    print(model, negs, epoch, getMAP('predsB/' + eval_file))


plt.style.use('scripts/rob.mplstyle')
fig, ax = plt.subplots(figsize=(8,5), dpi=300)
print(len(models))
    
for model in models:
    name = model.replace('multilingual.', '')[:-1]
    if name not in table_data:
        table_data[name] = {}
    table_data[name] = max(models[model])
    print(model, models[model].index(max(models[model])))
    ax.plot(range(1,7), models[model], label=model[model.find('.')+1:])

#leg = ax.legend(loc='lower right')
#leg.get_frame().set_linewidth(1.5)
    
ax.set_ylabel('MAP')
ax.set_xlabel('epoch')
fig.savefig('taskb.pdf', bbox_inches='tight')


for lm in table_data:
    scores = [lm]
    scores.append('{:.4f}'.format(table_data[lm]))
    print(' & '.join(scores) + ' \\\\')

