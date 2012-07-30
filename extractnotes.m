function extractnotes(inputvideo)
tic
vid_obj=mmreader(inputvideo);p=2;
grfram(:,:,1)=rgb2gray(read(vid_obj,1));colorframnum(1,1)=1;
step=1*vid_obj.FrameRate;
for i=2:step:vid_obj.NumberOfFrames
    temp1=rgb2gray(read(vid_obj,i));
    if (ysorno(grfram(:,:,p-1),temp1)==1)
        grfram(:,:,p)=rgb2gray(read(vid_obj,i));
        colorframnum(1,p)=i;
        p=p+1;
    end
end
takim(:,:,1)=uint8(zeros(size(grfram,1),size(grfram,2)));
for p=1:size(grfram,3)
    for i=1:size(grfram,1)
        for j=1:size(grfram,2)
            flag=0;
            for k=0:20:240
                if (abs(grfram(i,j,p)-k)<=10 && flag == 0)
                    takim(i,j,p)=uint8(k-1);
                    flag=1;
                end
            end
        end
    end
end
for t=1:size(grfram,3)
    [h go(t)]=max(imhist(takim(:,:,t)));
end
m=1;difframes=0;
for i=1:size(go,2)
    count=0;flag=0;
    for j=1:size(go,2)
        for k=1:size(difframes,2)
            if difframes(k)== go(i)
                flag=1;
            end
        end
        if (abs(go(i)-go(j))==0 && flag==0)
            count=count+1;
        end
    end
    if count>=2
    difframes(m)=go(i);
     inddifram(m)=i;
    m=m+1;
    end
end
for i=1:2
    t1= colorframnum(1,inddifram(i));
    inputforfacdetect=read(vid_obj,t1);
    oneintwo=face_detection1(inputforfacdetect);
    if oneintwo~=1
        reqframhist=difframes(i);
    end
end
m=1;
for i=1:size(grfram,3)
    if reqframhist-go(i)==0
      nameforim=[num2str(m) '.jpg'];
      imwrite(grfram(:,:,i),nameforim);
      m=m+1;
    end
end
function out=ysorno(imag1,imag2)
out=0;
cnt=0;
for i=1:size(imag1,1)
    for j=1:size(imag1,2)
        if(abs(imag1(i,j) - imag2(i,j)) > 40)
            cnt=cnt+1;
        end
    end
end
if(cnt > 200)
    out=1;
end
function out=face_detection1(imgfram)
out=0;
rmean=135.6252;bmean=101.6427;rbcov=[
  543.9556  322.5225
  322.5225  369.6283];
[likely_skin]=get_likelyhood(imgfram,rmean,bmean,rbcov);
[skinBW,opt_th] = segment_adaptive(likely_skin);
[erodedBW]=label_regions(skinBW);
[aspectBW]=aspect_test(erodedBW);
[templateBW]=template_test(aspectBW,imgfram,imgfram);
if sum(sum(templateBW))>50
    out=1;
end
function[likely_skin]=get_likelyhood(imgfram,rmean,bmean,rbcov)
img =imgfram;
imycbcr = rgb2ycbcr(img);
[m,n,l] = size(img);
likely_skin = zeros(m,n);
for i = 1:m
   for j = 1:n
      cr = double(imycbcr(i,j,3));
      cb = double(imycbcr(i,j,2));
      x = [(cr-rmean);(cb-bmean)];
      likely_skin(i,j) = [power(2*pi*power(det(rbcov),0.5),-1)]*exp(-0.5* x'*inv(rbcov)* x);
   end
end
lpf= 1/9*ones(3);
likely_skin = filter2(lpf,likely_skin);
likely_skin = likely_skin./max(max(likely_skin));
function [binary_skin,opt_th] = segment_adaptive(likely_skin)
[m,n] = size(likely_skin);
temp = zeros(m,n);
diff_list = [];
high=0.55;
low=0.01;
step_size=-0.1;
bias_factor=1;
indx_count=[(high-low)/abs(step_size)]+2;
for threshold = high:step_size:low
   binary_skin = zeros(m,n);
   binary_skin(find(likely_skin>threshold)) = 1;
   diff = sum(sum(binary_skin - temp));
   diff_list = [diff_list diff];
   temp = binary_skin;
end
[C, indx] = min(diff_list);
opt_th = (indx_count-indx)*abs(step_size)*bias_factor;
binary_skin = zeros(m,n);
binary_skin(find(likely_skin>opt_th)) = 1;
function[labelBW]=label_regions(binary_skin)
[m,n] = size(binary_skin); 
filledBW=zeros(m,n);
filledBW = imfill(binary_skin,'holes');
se2 = strel('disk',10); 
erodedBW=zeros(m,n);
erodedBW = imerode(filledBW,se2);
se1 = strel('disk',8); 
dilateBW=zeros(m,n);
dilateBW=imdilate(erodedBW,se1);
dilateBW = immultiply(dilateBW,binary_skin);
labelBW=zeros(m,n);
[labelBW,num] = bwlabel(dilateBW,8);
function [aspectBW]=aspect_test(eulerBW)
[m,n]=size(eulerBW);
filledBW = imfill(eulerBW,'holes');
se1 = strel('disk',3); 
growBW=zeros(m,n);
growBW=imdilate(filledBW,se1);
[labels,num] = bwlabel(growBW,8);
[aspect_ratio]=get_aspect(labels);
region_index = find(aspect_ratio<=3.5 & aspect_ratio>=1);
aspectBW=zeros(m,n);
for i=1:length(region_index)
    [x,y] = find(bwlabel(filledBW) == region_index(i));
    bwsegment = bwselect(filledBW,y,x,8);
    aspectBW=aspectBW+bwsegment;
end
function [ratiolist] = get_aspect(inputBW)
major = regionprops(inputBW,'MajorAxisLength');
major_length=cat(1,major.MajorAxisLength);
minor = regionprops(inputBW,'MinorAxisLength');
minor_length=cat(1,minor.MinorAxisLength);
ratiolist=major_length./minor_length;
function [template_passed]=template_test(aspectBW,imgfram,templat)
template(:,:,1)=rgb2gray(templat);
template(:,:,2)=rgb2gray(templat);
template(:,:,3)=rgb2gray(templat);
imgray=rgb2gray(imgfram);
imtemplate=template;
[labels,num] = bwlabel(aspectBW,8);
[m,n]=size(aspectBW);
orient = regionprops(labels,'Orientation');
angles=cat(1,orient.Orientation);
c = regionprops(labels,'Centroid');
centroids=cat(1,c.Centroid);
template_passed=zeros(m,n);
gray_matched=zeros(m,n);
for j=1:num,
[x,y] = find(labels == j);
bwsegment = bwselect(aspectBW,y,x,8);
oneface=immultiply(bwsegment,imgray);
cx1=centroids(j,1);
cy1=centroids(j,2);
p=regionprops(bwlabel(bwsegment),'BoundingBox');
boxdim=cat(1,p.BoundingBox);
regw=boxdim(3);
regh=boxdim(4);   
ratio=regh/regw;
if(ratio>1.6)
regh=1.5*regw;
cy1=cy1-(0.1*regh);        
end
gmodel_resize=imresize(imtemplate,[regh regw],'bilinear');
if(angles(j)>0)
gmodel_rotate=imrotate(gmodel_resize,angles(j)-90,'bilinear','loose');
else
gmodel_rotate=imrotate(gmodel_resize,90+angles(j),'bilinear','loose');
end
bwmodel=im2bw(gmodel_rotate,0);
[g,h]=size(bwmodel);
bwmorphed = bwmorph(bwmodel,'clean');
[L,no]=bwlabel(bwmorphed,8);
if(no==1)
bwsingle=bwmorphed;
else
ar=regionprops(bwlabel(bwmorphed),'Area');
areas=cat(1,ar.Area);  
[C,I]=max(areas);
[x1,y1] = find(bwlabel(bwmorphed)== I);
bwsingle = bwselect(bwmorphed,y1,x1,8);
end
filledmodel=regionprops(bwlabel(bwsingle),'FilledImage');
bwcrop=filledmodel.FilledImage;
[modh,modw]=size(bwcrop);
gmodel_crop=imresize(gmodel_rotate,[modh modw],'bilinear');
cenmod=regionprops(bwlabel(bwcrop),'Centroid');
central=cat(1,cenmod.Centroid);
cx2=central(1,1);
cy2=central(1,2);
mfit = zeros(size(oneface));
mfitbw = zeros(size(oneface));
[limy, limx] = size(mfit);
 startx = cx1-cx2;
 starty = cy1-cy2;
 endx = startx + modw-1;
 endy = starty + modh-1;	  
  startx = checklimit(startx,limx);
  starty = checklimit(starty,limy);
  endx = checklimit(endx,limx);
  endy = checklimit(endy,limy);
  for i=starty:endy,
   for j=startx:endx,
     mfit(round(i),round(j)) = gmodel_crop(round(i-starty+1),round(j-startx+1));
    end;
  end;
gray_matched=gray_matched+mfit;
crosscorr =corr2(mfit,oneface);
if(crosscorr>=0.6)    
    template_passed=template_passed+bwsegment;
end;
end;
function newcoord = checklimit(coord,maxval)
 newcoord = coord;
  if (newcoord<1) newcoord=1; end;
  if (newcoord>maxval) newcoord=maxval; end;
  toc