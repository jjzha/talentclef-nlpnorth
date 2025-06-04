import os
import json
import myutils
import _jsonnet

def load_json(path: str):
    return json.loads(_jsonnet.evaluate_snippet("", '\n'.join(open(path).readlines())))

def makeParams(defaultPath, mlm, lr = None, epochs = None):
    config = load_json(defaultPath)
    config['transformer_model'] = mlm
    config['training']['keep_top_n'] = 99
    tgt_path = 'configs/params.' + mlm.split('/')[-1] 
    if lr != None:
        config['training']['optimizer']['lr'] = lr
        tgt_path += '.' + str(lr)
    if epochs != None:
        config['training']['num_epochs'] = epochs
        config['training']['learning_rate_scheduler']['gradual_unfreezing'] = False
        tgt_path += '.' + str(epochs)
    tgt_path += '.json'

    if not os.path.isfile(tgt_path):
        json.dump(config, open(tgt_path, 'w'), indent=4)
    return tgt_path


for neg_sampling in [1, 2, 5]:#range(1,5):
    data_config_paths = []
    setup_name = 'taskb.train.neg' + str(neg_sampling)
    train_path = '../data/' + setup_name
    dev_path = '../data/taskb..dev'
    config = {}
    config['train_data_path'] = train_path
    #config['dev_data_path'] = dev_path
    config['sent_idxs'] = [0, 1]
    config['tasks'] = {'similar': {'column_idx': 2, 'task_type': 'classification'}}
    conf_path = 'configs/' + setup_name  + '.json'
    json.dump({setup_name: config}, open(conf_path, 'w'), indent=4)
    data_config_paths.append('../' + conf_path)
    
    for lm in myutils.llms + myutils.sllms:
        model_name = 'taskB.' + lm.split('/')[-1] + '.' + str(neg_sampling)
        #param_path = makeParams('machamp/configs/params.json', lm)
        #cmd = 'python3 train.py --dataset_configs ' + ' '.join(data_config_paths) + ' --parameters_config ../' + param_path.replace('machamp/', '') + ' --name ' + model_name
        #print(cmd)
        #if 'esco' in lm:
        #param_path = makeParams('machamp/configs/params.json', lm, lr=0.00001)
        #cmd = 'python3 train.py --dataset_configs ' + ' '.join(data_config_paths) + ' --parameters_config ../' + param_path.replace('machamp/', '') + ' --name ' + model_name + '-lowlr'
        #print(cmd)            
        param_path = makeParams('machamp/configs/params.json', lm, epochs=3)
        cmd = 'python3 train.py --dataset_configs ' + ' '.join(data_config_paths) + ' --parameters_config ../' + param_path.replace('machamp/', '') + ' --name ' + model_name + '-lessepochs3'
        if myutils.getModel(model_name + '-lessepochs3') == '':
            print(cmd)            



