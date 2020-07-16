import gensim
import shutil
import pickle

def getFileLineNums(filename):
    f=open(filename,'r')
    count=0
    for _ in f:
        count+=1
    return count

def prepend_line(infile,outfile,line):
    with open(infile,'r') as old:
        with open(outfile,'w') as new:
            new.write(str(line)+"\n")
            shutil.copyfileobj(old, new)

def load(filename):
    num_lines=getFileLineNums(filename)
    gensim_file='../data/query_feature/glove_model.txt'
    gensim_first_line="{} {}".format(num_lines,300)

    prepend_line(filename,gensim_file,gensim_first_line)

    gensim.models.KeyedVectors.load_word2vec_format(gensim_file)

load('../data/query_feature/glove.840B.300d.txt')


model=gensim.models.KeyedVectors.load_word2vec_format('../data/query_feature/glove_model.txt')

dictionary={}

with open("../data/origin_data/Dense_per_shot_tags/Dictionary.txt",'r') as f:
    for line in f:
        word=line.strip()[1:-1]
        if word not in model:
            if word=="Cupglass":
                dictionary["Glass"]=model["Glass"]
            if word=="Musicalinstrument":
                dictionary["Instrument"]=model["Instrument"]
            if word=="Petsanimal":
                dictionary["Animal"]=model["Animal"]
        else:
            dictionary[word]=model[word]


f=open("../data/query_feature/query_dictionary.pkl",'wb')
pickle.dump(dictionary,f)

