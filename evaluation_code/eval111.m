function score = eval111(sys_sum, gt_sum, vid_id)
%% load the dense annotations collected for videos in UT Egocentric dataset
% There are four videos in the dataset thats why loading the following .mat
% file will result in a structure of size four (4), each belonging to a
% video. The field "Bin_Vecs" of each structure contains one matrix, where
% each row is the bintary vector represents presence/absence of a concept in
% the shot. Consequently, the number of rows in he matrix shows the number
% of 5-second-long shots in the video.
load('Tags.mat');
%%

Bin_vecs = Tags(vid_id).Bin_Vecs;

sys_summary = Bin_vecs(sys_sum);
GT = Bin_vecs(gt_sum);

tmp = pdist2(cell2mat(sys_summary),cell2mat(GT),@mydistfunc);

%% The following function is borrowed from: 
% https://www.mathworks.com/matlabcentral/fileexchange/24134-gaimc---graph-algorithms-in-matlab-code?focused=5114622&tab=function
% It applies the maximum graph matching on the similarity matrix generated
% above
pre = bipartite_matching(tmp)/size(tmp,1); rec = bipartite_matching(tmp)/size(tmp,2);

score.pre = pre;
score.rec = rec;
score.f1 = 2*pre*rec/(pre+rec);
end

