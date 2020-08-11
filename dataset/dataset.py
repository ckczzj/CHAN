import os
import h5py
import torch
from torch.utils.data.dataset import Dataset
from utils import load_pickle

class UCTDataset(Dataset):
    def __init__(self,config):
        self.config=config
        self.dataset=[]
        for video_id in self.config["train_videos"]:
            for _ , _, files in os.walk("./data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
                for file in files:
                    self.dataset.append(file[:file.find("_oracle.txt")]+"_"+str(video_id))
        self.embedding=load_pickle("./data/processed/query_dictionary.pkl")

    def __getitem__(self,index):
        video_id=self.dataset[index].split('_')[2]
        f=h5py.File('./data/processed/V'+video_id+'_resnet_avg.h5','r')
        features=f['features'][()]
        seg_len=f['seg_len'][()]

        transfer={"Cupglass":"Glass",
                  "Musicalinstrument":"Instrument",
                  "Petsanimal":"Animal"}

        concept1,concept2=self.dataset[index].split('_')[0:2]

        concept1_GT=torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
        concept2_GT=torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"])
        with open("./data/origin_data/Dense_per_shot_tags/P0"+video_id+"/P0"+video_id+".txt","r") as f:
            lines=f.readlines()
            for index,line in enumerate(lines):
                concepts=line.strip().split(',')
                if concept1 in concepts:
                    concept1_GT[index]=1
                if concept2 in concepts:
                    concept2_GT[index]=1

        shot_num=seg_len.sum()
        mask_GT=torch.zeros(self.config["max_segment_num"]*self.config["max_frame_num"],dtype=torch.bool)
        for i in range(shot_num):
            mask_GT[i]=1

        if concept1 in transfer:
            concept1=transfer[concept1]
        if concept2 in transfer:
            concept2=transfer[concept2]
        concept1=self.embedding[concept1]
        concept2=self.embedding[concept2]

        return features,seg_len,concept1,concept2,concept1_GT,concept2_GT,mask_GT

    def __len__(self):
        return len(self.dataset)