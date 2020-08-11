import json
import pickle

def load_json(filename):
    with open(filename, encoding='utf8') as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def save_pickle(object,filename):
    with open(filename, 'wb') as f:
        pickle.dump(object,f)
