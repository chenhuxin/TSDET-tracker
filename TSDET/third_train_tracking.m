function net_online3 =third_train_tracking(opts,varargin,config)
global objSize;
num_channels=64;

% training options (SGD)
opts.train = struct([]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;
output_sigma_factor = 0.1;

[net1,avgImg]=initVGG16Net();
[Gt]=config.gt;

% I only have a Titan Black GPU on my side. Its memory is limited.
% For objects with large size, I have to perform the downsampling.
% You can remove the downsampling on your side. The results might be little different.
scale=1;
global resize;
objSize=Gt(1,3:4);
if objSize(1)*objSize(2)>resize*resize
    scale=resize/max(objSize);    
    disp('resized');
end

im1=imread(config.imgList{1});%360*640*3
im=imresize(im1,scale);
cell_size=4;
if size(im,3)==1
    im = cat(3, im, im, im);
    im1 = cat(3, im1, im1, im1);
end

targetLoc=round(Gt(1,:)*scale);%[446 181 29 26]
target_sz=[targetLoc(4) targetLoc(3)];
im_sz=size(im);
window_sz = get_search_window(target_sz, im_sz);%[145 145]
l1_patch_num = ceil(window_sz/ cell_size);
l1_patch_num=l1_patch_num-mod(l1_patch_num,2)+1;
cos_window = hann(l1_patch_num(1)) * hann(l1_patch_num(2))';%37*37

sz_window=size(cos_window);
pos = [targetLoc(2), targetLoc(1)] + floor(target_sz/2);%第一帧的目标中心位置
patch = get_subwindow(im, pos, window_sz);%145*145*3
meanImg=zeros(size(patch));
meanImg(:,:,1)=avgImg(1);
meanImg(:,:,2)=avgImg(2);
meanImg(:,:,3)=avgImg(3);
patch1 = single(patch) - meanImg;
net1.conserveMemory=false;
net1.eval({'input',gpuArray(patch1)});

index=[17];%conv3
feat=cell(length(index),1);
for i=1:length(index)
    feat1 = gather(net1.vars(index(i)).value);%37*37*512
    feat1 = imResample(feat1, sz_window(1:2));%37*37*512
    feat{i} = bsxfun(@times, feat1, cos_window);                        
end
feat=feat{1};

[hf,wf,cf]=size(feat);
matrix=reshape(feat,hf*wf,cf);%1369*512
coeff = pca(matrix);%512*512 coeff主成分系数 应该就是协方差矩阵的特征向量矩阵(也就是映射矩阵).完整输出的情况下是一个p-by-p 的matrix.每一列都是一个特征向量.按对应的特征值
coeff=coeff(:,1:num_channels);%512*64

target_sz1=ceil(target_sz/cell_size);%[7 8]
output_sigma = target_sz1*output_sigma_factor;
label=gaussian_shaped_labels(output_sigma, l1_patch_num);%37*37

%-------------------first frame initialization-----------
trainOpts.numEpochs=1000;

feat_=reshape(feat,hf*wf,cf);
feat_=feat_*coeff;%pca降维之后的feat 1396*64
featPCA=reshape(feat_,hf,wf,num_channels);%37*37*64

net_online3=initNet(target_sz1);

trainOpts.batchSize = 1 ;
trainOpts.numSubBatches = 1 ;
trainOpts.continue = true ;
trainOpts.gpus =1;
trainOpts.prefetch = true ;

trainOpts.expDir = opts.expDir ;
trainOpts.learningRate=5e-8;
trainOpts.weightDecay= 1;

train=1;
imdb=[];
input={featPCA label};
opts.train.gpus=1;
bopts.useGpu = numel(opts.train.gpus) > 0 ;
net_online3 = cnn_train_dag(net_online3, imdb, input,getBatchWrapper(bopts), ...%用dagnn框架训练网络
                     trainOpts, ...
                     'train', train, ...                     
                     opts.train) ;
net_online3.move('gpu');
end
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,false,opts,'prefetch',nargout==0) ;
end