function dist2 = mydistfunc(v1,v2)
tmp2 = v2.*repmat(v1,size(v2,1),1);
dist2 = sum(tmp2,2)./(sum(v1,2)+sum(v2,2)-sum(tmp2,2));
dist2(isnan(dist2)) = 0;
end
