import h5py
import torch as t
import numpy as np
from segment import cpd_auto
from utils import load_json

config=load_json("../config/config.json")

for kind in ["C3D","resnet_avg"]:
    print(kind)
    for video_id in ["V1","V2","V3","V4"]:
        f=h5py.File('../data/features/'+video_id+'_'+kind+'.h5','r')
        feature=f['feature'][()]
        frame_num=feature.shape[0]
        print(frame_num)

        K=feature
        K=np.dot(K,K.T)

        cps,_=cpd_auto(K,config["max_segment_num"]-1,1,desc_rate=1,verbose=False,lmax=config["max_frame_num"]-1) #int(K.shape[0]/25)
        seg_num=len(cps)+1

        assert seg_num<=config["max_segment_num"]

        seg_points=cps
        seg_points=np.insert(seg_points,0,0)
        seg_points=np.append(seg_points,frame_num)

        segments=[]
        for i in range(seg_num):
            segments.append(np.arange(seg_points[i],seg_points[i+1],1,dtype=np.int32))

        assert len(segments)<=config["max_segment_num"]

        for seg in segments:
            assert len(seg)<=config["max_frame_num"]

        seg_len=np.zeros((config["max_segment_num"]),dtype=np.int32)
        for index,seg in enumerate(segments):
            seg_len[index]=len(seg)

        # features

        for seg in segments:
            for frame in seg:
                assert frame<frame_num

        features=t.zeros((config["max_segment_num"],config["max_frame_num"],4096 if kind=="C3D" else 2048))
        for seg_index,seg in enumerate(segments):
            for frame_index,frame in enumerate(seg):
                features[seg_index,frame_index]=t.tensor(feature[frame])
                # features[seg_index,frame_index]=F.avg_pool1d(t.tensor(feature[frame]).unsqueeze(0).unsqueeze(0),kernel_size=2,stride=2)


        f=h5py.File('../data/processed/'+video_id+'_'+kind+'.h5','w')
        f.create_dataset('features', data=features)
        f.create_dataset('seg_len', data=seg_len)

        f.close()