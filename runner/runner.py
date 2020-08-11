import os
import h5py
import torch
from torch.utils.data import DataLoader
import matlab.engine

from model import CHAN
from dataset import UCTDataset
from utils import load_pickle


class Runner():
    def __init__(self,config):
        self.config=config
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config["gpu"]
        self._build_dataloader()
        self._bulid_model()
        self._build_optimizer()
        self.max_f1=0
        self.max_p=0
        self.max_r=0


    def _bulid_model(self):
        self.model = CHAN(self.config).cuda()

    def _build_dataset(self):
        return UCTDataset(self.config)

    def _build_dataloader(self):
        dataset=self._build_dataset()
        self.dataloader=DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["num_workers"])

    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def output(self):
        print(" max_p = ",self.max_p," max_r = ",self.max_r," max_f1 = ",self.max_f1)


    def train(self):
        print("start to evaluate random result")
        self.evaluate(self.config["test_video"],self.config["top_percent"])
        print("end to evaluate random result")

        criterion=torch.nn.BCELoss()

        for epoch in range(self.config["epoch"]):
            batch_count=0
            for features,seg_len,concept1,concept2,concept1_GT,concept2_GT,mask_GT in self.dataloader:
                train_num=seg_len.shape[0]
                batch_count+=1
                self.optimizer.zero_grad()

                mask=torch.zeros(train_num,self.config["max_segment_num"],self.config["max_frame_num"],dtype=torch.bool).cuda()
                for i in range(train_num):
                    for j in range(len(seg_len[i])):
                        for k in range(seg_len[i][j]):
                            mask[i][j][k]=1

                # batch_size * max_seg_num * max_seg_length
                concept1_score,concept2_score=self.model(features.cuda(),seg_len.cuda(),concept1.cuda(),concept2.cuda())

                loss=torch.zeros(1).cuda()
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
                self.optimizer.step()

            self.evaluate(self.config["test_video"],self.config["top_percent"])

    def evaluate(self,video_id,top_percent):
        current_work_dir = os.getcwd()
        os.chdir("./evaluation_code")
        evaluator = matlab.engine.start_matlab()
        os.chdir(current_work_dir)

        f1=0
        p=0
        r=0

        embedding=load_pickle("./data/processed/query_dictionary.pkl")

        evaluation_num=0

        for _,_,files in os.walk("./data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)):
            evaluation_num=len(files)
            for file in files:
                summaries_GT=[]
                with open("./data/origin_data/Query-Focused_Summaries/Oracle_Summaries/P0"+str(video_id)+"/"+file,"r") as f:
                    for line in f.readlines():
                        summaries_GT.append(int(line.strip()))
                f=h5py.File('./data/processed/V'+str(video_id)+'_resnet_avg.h5','r')
                features=torch.tensor(f['features'][()]).unsqueeze(0).cuda()
                seg_len=torch.tensor(f['seg_len'][()]).unsqueeze(0).cuda()

                transfer={"Cupglass":"Glass","Musicalinstrument":"Instrument","Petsanimal":"Animal"}

                concept1,concept2=file.split('_')[0:2]

                if concept1 in transfer:
                    concept1=transfer[concept1]
                if concept2 in transfer:
                    concept2=transfer[concept2]

                concept1=torch.tensor(embedding[concept1]).unsqueeze(0).cuda()
                concept2=torch.tensor(embedding[concept2]).unsqueeze(0).cuda()

                mask=torch.zeros(1,self.config["max_segment_num"],self.config["max_frame_num"],dtype=torch.bool).cuda()
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

