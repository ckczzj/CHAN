%% Video ID
% This is used to identify the video you are trying to evaluate summaries
% for. 
vid_id = 2;

%% Generate two random summaries for testing purposes
sys_sum = randi(3600,1,150);
gt_sum = randi(3600,1,100);

%% Compute Precision/Recall/F1 scores
score = eval111(sys_sum,gt_sum,vid_id)

