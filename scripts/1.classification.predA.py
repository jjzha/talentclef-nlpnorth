import os
import myutils
import json

for model_name in os.listdir('machamp/logs/'):
    if 'taskB' in model_name:
        continue
    main_model_path = myutils.getModel(model_name)
    if main_model_path == '':
        continue
    model_root = main_model_path.replace('.pt', '_')
    for i in range(1,21):
        model_path = model_root + str(i) + '.pt'
        if not os.path.isfile(model_path):
            continue
        paths = []
        for language in ['english', 'german', 'spanish', 'chinese']:
            dev_path =  '../data/taska.' + language + '.dev'
            out_path = '../preds/' + model_name + '.' + language + '.' + str(i) 
            if not os.path.isfile(out_path[3:]):
                paths.append(dev_path)
                paths.append(out_path)
        data_config_path = main_model_path.replace('model.pt', 'dataset-configs.json')
        dataset_name = [x for x in json.load(open(data_config_path))][0]

        if len(paths) != 0:
            cmd = 'python3 predict.py ' + model_path.replace('machamp/', '') + ' ' + ' '.join(paths) + '  --topn 99 --dataset ' + dataset_name 
            print(cmd)

