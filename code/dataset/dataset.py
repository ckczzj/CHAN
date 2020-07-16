from torch.utils.data.dataset import Dataset
import os
import h5py
import pickle
import torch as t
from config.config import DefaultConfig as config

class UCTDataset(Dataset):
    def __init__(self,videos_id):
        self.dataset=[]
        for video_id in videos_id:
            for _ , _, files in os.walk("../../data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
                for file in files:
                    self.dataset.append(file[:file.find("_oracle.txt")]+"_"+str(video_id))
        with open("../../data/query_feature/query_dictionary.pkl","rb") as f:
            self.embedding=pickle.load(f)

    def __getitem__(self,index):
        video_id=self.dataset[index].split('_')[2]
        f=h5py.File('../../data/riple_data/V'+video_id+'_resnet_avg.h5','r')
        features=f['features'][()]
        seg_len=f['seg_len'][()]

        transfer={"Cupglass":"Glass",
                  "Musicalinstrument":"Instrument",
                  "Petsanimal":"Animal"}

        concept1,concept2=self.dataset[index].split('_')[0:2]

        concept1_GT=t.zeros(config.MAX_SEGMENT_NUM*config.MAX_FRAME_NUM)
        concept2_GT=t.zeros(config.MAX_SEGMENT_NUM*config.MAX_FRAME_NUM)
        with open("../../data/origin_data/Dense_per_shot_tags/P0"+video_id+"/P0"+video_id+".txt","r") as f:
            lines=f.readlines()
            for index,line in enumerate(lines):
                concepts=line.strip().split(',')
                if concept1 in concepts:
                    concept1_GT[index]=1
                if concept2 in concepts:
                    concept2_GT[index]=1

        shot_num=seg_len.sum()
        mask_GT=t.zeros(config.MAX_SEGMENT_NUM*config.MAX_FRAME_NUM,dtype=t.uint8)
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

# if __name__=="__main__":
    # train_dataloader=DataLoader(UCTDataset([4]),batch_size=1,shuffle=True,num_workers=5)
    #
    # for f,sn,sl,q1,q2,c1,c2 in train_dataloader:
    #     # print(f,sn,sl,q1,q2)
    #     # print(f.shape,sn.shape,sl.shape,q1.shape,q2.shape)
    #     print(sl.shape)
    #     break