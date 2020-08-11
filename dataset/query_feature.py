from utils import save_pickle
from gensim.models import KeyedVectors


word2vec=KeyedVectors.load_word2vec_format('../data/glove.bin', binary=True)

dictionary={}

with open("../data/origin_data/Dense_per_shot_tags/Dictionary.txt",'r') as f:
    for line in f:
        word=line.strip()[1:-1]
        if word not in word2vec:
            if word=="Cupglass":
                dictionary["Glass"]=word2vec["Glass"]
            if word=="Musicalinstrument":
                dictionary["Instrument"]=word2vec["Instrument"]
            if word=="Petsanimal":
                dictionary["Animal"]=word2vec["Animal"]
        else:
            dictionary[word]=word2vec[word]

save_pickle(dictionary,"../data/processed/query_dictionary.pkl")

