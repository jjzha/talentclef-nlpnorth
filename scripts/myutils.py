import os


#llms = ['jjzha/esco-xlm-roberta-large', 'microsoft/mdeberta-v3-base', 'studio-ousia/mluke-large', 'Twitter/twhin-bert-large', 'microsoft/infoxlm-large', 'xlm-roberta-large', 'Alibaba-NLP/gte-Qwen2-1.5B-instruct', 'Alibaba-NLP/gte-Qwen2-7B-instruct']
llms = ['jjzha/esco-xlm-roberta-large', 'microsoft/mdeberta-v3-base', 'studio-ousia/mluke-large', 'microsoft/infoxlm-large', 'xlm-roberta-large']

first_llm = llms[0]

sllms = ['BAAI/bge-m3', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'intfloat/multilingual-e5-large-instruct', 'intfloat/multilingual-e5-large', 'ibm-granite/granite-embedding-278m-multilingual', 'sentence-transformers/LaBSE']


first_sllm = sllms[:2]

train_langs = ['english', 'german', 'spanish']


def getModel(name):
    modelDir = 'machamp/logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.pt'
            if os.path.isfile(modelPath):
                return modelPath
    return ''

