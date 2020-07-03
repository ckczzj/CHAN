from model.CHAN import CHAN
from model.CHAN_without_SA import CHAN_without_SA
from model.CHAN_without_GA import CHAN_without_GA
import torch as t
from dataset.dataset import UCTDataset
from torch.utils.data.dataloader import DataLoader
import pickle
import h5py
import matlab.engine
from config.config import DefaultConfig as config

import os
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

class Trainer():
    def __init__(self):
        self.max_f1=0
        self.max_p=0
        self.max_r=0
        self.model=CHAN().cuda()
        # self.model=CHAN_without_SA().cuda()
        # self.model=CHAN_without_GA().cuda()
        print("CHAN",config.train_videos,config.test_video,config.TOP_PERCENT)

    def output(self):
        print(" max_p = ",self.max_p," max_r = ",self.max_r," max_f1 = ",self.max_f1)


    def train(self,train_videos_id,test_video_id,top_percent=0.04,batch_size=1,num_workers=5):
        print("random")
        self.evaluate(test_video_id,top_percent)
        print("random")

        train_data=UCTDataset(train_videos_id)
        train_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers)

        criterion=t.nn.BCELoss()

        optimizer=t.optim.Adam(self.model.parameters(),lr=config.lr)

        for epoch in range(config.MAX_EPOCH):
            batch_count=0
            for features,seg_len,concept1,concept2,concept1_GT,concept2_GT,mask_GT in train_dataloader:
                train_num=seg_len.shape[0]
                batch_count+=1
                optimizer.zero_grad()

                mask=t.zeros(train_num,config.MAX_SEGMENT_NUM,config.MAX_FRAME_NUM,dtype=t.uint8).cuda()
                for i in range(train_num):
                    for j in range(len(seg_len[i])):
                        for k in range(seg_len[i][j]):
                            mask[i][j][k]=1

                # batch_size * max_seg_num * max_seg_length
                concept1_score,concept2_score=self.model(features.cuda(),seg_len.cuda(),concept1.cuda(),concept2.cuda())

                loss=t.zeros(1).cuda()
                for i in range(train_num):
                    concept1_score_tmp=concept1_score[i].masked_select(mask[i]).unsqueeze(0)
                    concept2_score_tmp=concept2_score[i].masked_select(mask[i]).unsqueeze(0)
                    concept1_GT_tmp=concept1_GT[i].masked_select(mask_GT[i]).unsqueeze(0)
                    concept2_GT_tmp=concept2_GT[i].masked_select(mask_GT[i]).unsqueeze(0)

                    loss1=criterion(concept1_score_tmp,concept1_GT_tmp.cuda())
                    loss2=criterion(concept2_score_tmp,concept2_GT_tmp.cuda())
                    loss+=loss1+loss2

                if (batch_count+1)%5==0:
                    print("epoch ",epoch+1," batch ",batch_count+1," loss ",loss.item()/train_num)
                loss.backward()
                optimizer.step()

            self.evaluate(test_video_id,top_percent)

    def evaluate(self,video_id,top_percent):
        current_work_dir = os.getcwd()
        os.chdir("../evaluation_code")
        evaluator = matlab.engine.start_matlab()
        os.chdir(current_work_dir)

        f1=0
        p=0
        r=0

        with open("../../data/query_feature/query_dictionary.pkl","rb") as f:
            embedding=pickle.load(f)

        evaluation_num=0

        for _,_,files in os.walk("../../data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
            evaluation_num=len(files)
            for file in files:
                summaries_GT=[]
                with open("../../data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)+"/"+file,"r") as f:
                    for line in f.readlines():
                        summaries_GT.append(int(line.strip()))
                f=h5py.File('../../data/riple_data/V'+str(video_id)+'_resnet_avg.h5','r')
                features=t.tensor(f['features'][()]).unsqueeze(0).cuda()
                seg_len=t.tensor(f['seg_len'][()]).unsqueeze(0).cuda()

                transfer={"Cupglass":"Glass","Musicalinstrument":"Instrument","Petsanimal":"Animal"}

                concept1,concept2=file.split('_')[0:2]

                if concept1 in transfer:
                    concept1=transfer[concept1]
                if concept2 in transfer:
                    concept2=transfer[concept2]

                concept1=t.tensor(embedding[concept1]).unsqueeze(0).cuda()
                concept2=t.tensor(embedding[concept2]).unsqueeze(0).cuda()

                mask=t.zeros(1,config.MAX_SEGMENT_NUM,config.MAX_FRAME_NUM,dtype=t.uint8).cuda()
                for i in range(1):
                    for j in range(len(seg_len[i])):
                        for k in range(seg_len[i][j]):
                            mask[i][j][k]=1
                concept1_score,concept2_score=self.model(features,seg_len,concept1,concept2)

                score=concept1_score+concept2_score

                score=score.masked_select(mask)

                _,top_index=score.topk(int(score.shape[0]*top_percent))

                top_index+=1

                out=evaluator.eval111(matlab.int32(list(top_index.cpu().numpy())),matlab.int32(summaries_GT),video_id)

                f1+=out["f1"]
                r+=out["rec"]
                p+=out["pre"]

        if f1/evaluation_num>self.max_f1:
            self.max_f1=f1/evaluation_num
            self.max_p=p/evaluation_num
            self.max_r=r/evaluation_num

        print("p = ",p/evaluation_num," r = ",r/evaluation_num," f1 = ",f1/evaluation_num)


if __name__=="__main__":

    trainer=Trainer()
    trainer.train(config.train_videos,config.test_video,top_percent=config.TOP_PERCENT,batch_size=config.BATCH_SIZE)
    trainer.output()

