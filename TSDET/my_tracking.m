function [result] = my_tracking(opts, varargin, config, display)

net_online1=first_train_tracking(opts,varargin,config); %第29层conv5
net_online2=second_train_tracking(opts,varargin,config);%第23层conv4
net_online3=third_train_tracking(opts,varargin,config); %第17层conv3
global objSize;
A=0.011;
num_channels=64;

opts.train = struct([]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;
output_sigma_factor = 0.1;

[net1,avgImg]=initVGG16Net();

nFrame=config.nFrames;
[Gt]=config.gt;
result = zeros(length(nFrame), 4);
result(1,:) = Gt(1,:);

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

index=[28,23,17];
featPCA1st=cell(length(index),1);
coeff=cell(length(index),1);
%-------------------first frame initialization-----------
for i=1:length(index)
    feat1 = gather(net1.vars(index(i)).value);%37*37*512
    feat1 = imResample(feat1, sz_window(1:2));%37*37*512
    feat= bsxfun(@times, feat1, cos_window);  
    [hf,wf,cf]=size(feat);
    feat_=reshape(feat,hf*wf,cf);%1369*512
    coeffient= pca(feat_);%512*512 coeff主成分系数 应该就是协方差矩阵的特征向量矩阵(也就是映射矩阵).完整输出的情况下是一个p-by-p 的matrix.每一列都是一个特征向量.按对应的特征值
    coeff{i}=coeffient(:,1:num_channels);%512*64
    feat_=feat_*coeff{i};%pca降维之后的feat 1396*64
    featPCA=reshape(feat_,hf,wf,num_channels);%37*37*64
    featPCA1st{i}=featPCA;
end

target_sz1=ceil(target_sz/cell_size);%[7 8]
output_sigma = target_sz1*output_sigma_factor;
label=gaussian_shaped_labels(output_sigma, l1_patch_num);%37*37

imd=[im1];
%-------------------Display First frame----------
if display    
    figure(1);
    set(gcf,'Position',[200 300 480 320],'MenuBar','none','ToolBar','none');
    hd = imshow(imd,'initialmagnification','fit'); hold on;
    rectangle('Position', Gt(1,:), 'EdgeColor', [1 0 0], 'Linewidth', 2);    
    set(gca,'position',[0 0 1 1]);
    text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30);
    hold off;
    drawnow;   
end
%----------------online prediction------------------
motion_sigma_factor=0.6;
cell_size=4;
global num_update;
num_update=10;%10帧更新一次
cur=1;
feat_update1=cell(num_update,1);
feat_update2=cell(num_update,1);
feat_update3=cell(num_update,1);
label_update=cell(num_update,1);
feat_preupdate1=cell(num_update,1);
feat_preupdate2=cell(num_update,1);
feat_preupdate3=cell(num_update,1);
target_szU=target_sz;
motion_sigma = target_sz1*motion_sigma_factor;    
motion_map=gaussian_shaped_labels(motion_sigma, l1_patch_num);    
for i=2:nFrame 
%______________前一帧的特征_____________
    im2=imread(config.imgList{i-1});
    img=imresize(im2,scale);
    if size(img,3)==1
        img=cat(3,img,img,img);
        im2=cat(3,im2,im2,im2);
    end
    patch_pre=get_subwindow(img,pos,window_sz);
    patch_pre1=single(patch_pre)-meanImg;
    net1.eval({'input',gpuArray(patch_pre1)});
    featPCA_pre=cell(length(index),1);
    for n=1:length(index)
        feat1_pre=gather(net1.vars(index(n)).value);
        feat1_pre=imResample(feat1_pre,sz_window(1:2));
        feat_pre=bsxfun(@times,feat1_pre,cos_window);
        [hf,wf,cf]=size(feat_pre);
        feature_pre=reshape(feat_pre,hf*wf,cf);
        feature_pre=feature_pre*coeff{n};
        featPCA_pre{n}=reshape(feature_pre,hf,wf,num_channels);
    end
%_________________当前帧的特征_____________________
    im1=imread(config.imgList{i});
    im=imresize(im1,scale);
    if size(im,3)==1
        im = cat(3, im, im, im);
        im1 = cat(3, im1, im1, im1);
    end
    patch = get_subwindow(im, pos, window_sz); %上一帧的pos      
    patch1 = single(patch) - meanImg;    
    net1.eval({'input',gpuArray(patch1)});
    maxres=zeros(length(index),1);
    response_layer=cell(length(index),1);
    experts=cell(length(index),1);
    featPCA=cell(length(index),1);
    for j=1:length(index)
        feat1 = gather(net1.vars(index(j)).value);
        feat1 = imResample(feat1, sz_window(1:2));
        feat= bsxfun(@times, feat1, cos_window); %消除边界效应  
        [hf,wf,cf]=size(feat) ;   
        feat_=reshape(feat,hf*wf,cf);
        feat_=feat_*coeff{j};
        featPCA{j}=reshape(feat_,hf,wf,num_channels); 
        
        if j==1
            net_online1.eval({'input1',gpuArray(featPCA{j}),'input2',gpuArray(featPCA_pre{1})});%input1:每一帧经过con5之后的特征 input2:前一帧的特征    
            regression_map=gather(net_online1.vars(10).value);  %输出响应图           
            response_layer{1}=regression_map.*motion_map;%使得远离中心点的响应权重变小 37*37
            maxres(1)=max(response_layer{1}(:));
            [row,col]=find(response_layer{1}==maxres(1),1);
            experts{1}.row=row;
            experts{1}.col=col;
        end
        if j==2
            net_online2.eval({'input1',gpuArray(featPCA{j}),'input2',gpuArray(featPCA_pre{2})});%每一帧经过con4之后的特征 input2:前一帧的特征    
            regression_map=gather(net_online2.vars(10).value);  %输出响应图           
            response_layer{2}=regression_map.*motion_map;%使得远离中心点的响应权重变小 37*37
            maxres(2)=max(response_layer{2}(:));
            [row,col]=find(response_layer{2}==maxres(2),1);
            experts{2}.row=row;
            experts{2}.col=col;
        end
        if j==3
            net_online3.eval({'input1',gpuArray(featPCA{j}),'input2',gpuArray(featPCA_pre{3})});%每一帧经过con3之后的特征 input2:前一帧的特征    
            regression_map=gather(net_online3.vars(10).value); %输出响应图           
            response_layer{3}=regression_map.*motion_map;%使得远离中心点的响应权重变小 37*37
%             figure(3);
%             surf(response_layer{3});
%             set(gca,'xtick',[],'xticklabel',[])
%             set(gca,'ytick',[],'yticklabel',[])
%             set(gca,'ztick',[],'zticklabel',[])
            maxres(3)=max(response_layer{3}(:));
            [row,col]=find(response_layer{3}==maxres(3),1);
            experts{3}.row=row;
            experts{3}.col=col;
        end
    end
    if i==2
        rect=round(Gt(2,:)*scale);
        pos2(1) = rect(2) + floor(rect(4)/2);
        pos2(2) = rect(1) + floor(rect(3)/2);
        row = (pos2(1) - pos(1,1))/cell_size+ceil(hf/2);
        col = (pos2(2) - pos(1,2))/cell_size+ceil(wf/2);
        W = [1 0.2 0.01];
        R(1:length(index)) = 0;
        loss(1:3,1:length(index)) = 0;
    else
        row = 0; col = 0;
        for jj = 1:length(index)
            row = row + W(jj)*experts{jj}.row;
            col = col + W(jj)*experts{jj}.col;
        end
    end
    vert_delta=ceil(row); horiz_delta=ceil(col);
    vert_delta  = vert_delta  - ceil(hf/2);
    horiz_delta = horiz_delta - ceil(wf/2);   
              
    pos = pos + cell_size * [vert_delta, horiz_delta];
               
    target_szU=scale_estimation(im,pos,target_szU,window_sz,...
            net1,net_online2,coeff{2},meanImg,featPCA1st{2});            
                            
    targetLoc=[pos([2,1]) - target_szU([2,1])/2, target_szU([2,1])];  
    if i==2
        result(i,:)=round(rect/scale); 
    else
        result(i,:)=round(targetLoc/scale);
    end
    imd=[im1];
%  -----------Display current frame-----------------
    if display   
        hc = get(gca, 'Children'); delete(hc(1:end-1));
        set(hd,'cdata',imd); hold on;                                
        rectangle('Position', result(i,:), 'EdgeColor', [1 0 0], 'Linewidth', 2);                       
        set(gca,'position',[0 0 1 1]);
        text(10,10,num2str(i),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
        hold off;
        drawnow;  
    end
%-------------------------------updata weights with hedge-------------------------
    row = round(row); col = round(col);
    
    for ii = 1:length(index)
        loss(3,ii) = maxres(ii)-response_layer{ii}(row,col);%l_t^k
    end
    lossA = sum(W.*loss(3,:));
    LosIdx = mod(i-1,2)+1;
    
    LosMean = mean(loss(1:2,:));%loss的均值
    LosStd = std(loss(1:2,:));%loss的方差
    LosMean(LosMean<0.0001) = 0;
    LosStd(LosStd<0.0001) = 0;
    
    curDiff = loss(3,:)-LosMean;
    alpha=0.97*exp((-10*abs(curDiff)./(LosStd+eps)));%alpha_t^k
    
    % Truncation
    alpha(alpha>0.9)=0.97;%稳定
    alpha(alpha<0.12)=0.119;%不稳定
    
    R=R.*(1-alpha)+alpha.*(lossA-loss(3,:));
    
    % Update loss history
    loss(LosIdx,:)=loss(3,:);
    
    c = find_nh_scale(R, A);
    W = nnhedge_weights(R, c, A);
    W = W / sum(W);    
    threshold=0.119;
 %-----------Model update------------------------- 
    if cur~=num_update
        if(alpha(1)==threshold)
            labelA=circshift(label,[vert_delta,horiz_delta]);
            trainOpts.batchSize = 1 ;
            trainOpts.numSubBatches = 1 ;
            trainOpts.continue = true ;
            trainOpts.gpus = 1 ;
            trainOpts.prefetch = true ;
            trainOpts.expDir = 'exp/update/' ;
            trainOpts.learningRate = 2e-9;
            trainOpts.weightDecay= 1;
            trainOpts.numEpochs = 2;
            train=1;
            imdb=[];
            inputs1={featPCA{1} featPCA_pre{1} labelA};
            opts.train.gpus=1;
            bopts.useGpu = numel(opts.train.gpus) > 0 ;
            
            info = cnn_train_dagnn(net_online1, imdb, inputs1,getBatchWrapper(bopts), ...
                trainOpts, ...
                'train', train, ...
                opts.train);
            net_online1.move('gpu');  
        end
        if(alpha(2)==threshold)
            labelB=circshift(label,[vert_delta,horiz_delta]);
            trainOpts.batchSize = 1 ;
            trainOpts.numSubBatches = 1 ;
            trainOpts.continue = true ;
            trainOpts.gpus = 1 ;
            trainOpts.prefetch = true ;
            trainOpts.expDir = 'exp/update/' ;
            trainOpts.learningRate = 2e-9;
            trainOpts.weightDecay= 1;
            trainOpts.numEpochs = 2;
            train=1;
            imdb=[];
            inputs2={featPCA{2} featPCA_pre{2} labelB};
            opts.train.gpus=1;
            bopts.useGpu = numel(opts.train.gpus) > 0 ;
            
            info = cnn_train_dagnn(net_online2, imdb, inputs2,getBatchWrapper(bopts), ...
                trainOpts, ...
                'train', train, ...
                opts.train);
            net_online2.move('gpu');  
        end
        if(alpha(3)==threshold)
            labelC=circshift(label,[vert_delta,horiz_delta]);
            trainOpts.batchSize = 1 ;
            trainOpts.numSubBatches = 1 ;
            trainOpts.continue = true ;
            trainOpts.gpus = 1;
            trainOpts.prefetch = true ;
            trainOpts.expDir = 'exp/update/' ;
            trainOpts.learningRate = 2e-9;
            trainOpts.weightDecay= 1;
            trainOpts.numEpochs = 2;
            train=1;
            imdb=[];
            inputs3={featPCA{3} featPCA_pre{3} labelC};
            opts.train.gpus=1;
            bopts.useGpu = numel(opts.train.gpus) > 0 ;
            
            info = cnn_train_dagnn(net_online3, imdb, inputs3,getBatchWrapper(bopts), ...
                trainOpts, ...
                'train', train, ...
                opts.train);
            net_online3.move('gpu');  
        end
    end
 
    labelU=circshift(label,[vert_delta,horiz_delta]);
    label_update{cur}=labelU;
    feat_update1{cur}=featPCA{1};%feat_update{1}=前一帧的feature feat_update{2}=现在帧的feature
    feat_preupdate1{cur}=featPCA_pre{1};
    if cur==num_update    

        trainOpts.batchSize = 1 ;
        trainOpts.numSubBatches = 1 ;
        trainOpts.continue = true ;
        trainOpts.gpus = 1;
        trainOpts.prefetch = true ;

        trainOpts.expDir = 'exp/update/' ;
        trainOpts.learningRate = 2e-9;        
        trainOpts.weightDecay= 1;
        trainOpts.numEpochs = 2;

        train=1;
        imdb=[];
        input1={feat_update1 feat_preupdate1 label_update};
        opts.train.gpus=1;
        bopts.useGpu = numel(opts.train.gpus) > 0 ;

        info = cnn_train_dag_update(net_online1, imdb, input1,getBatchWrapper(bopts), ...
                             trainOpts, ...
                             'train', train, ...                     
                             opts.train) ; 
        net_online1.move('gpu');  
    end
    feat_update2{cur}=featPCA{2}; %feat_update{1}=前一帧的feature feat_update{2}=现在帧的feature
    feat_preupdate2{cur}=featPCA_pre{2};
    if cur==num_update
         
         trainOpts.batchSize = 1 ;
         trainOpts.numSubBatches = 1 ;
         trainOpts.continue = true ;
         trainOpts.gpus =1 ;
         trainOpts.prefetch = true ;
         
         trainOpts.expDir = 'exp/update/' ;
         trainOpts.learningRate = 2e-9;
         trainOpts.weightDecay= 1;
         trainOpts.numEpochs = 2;
         
         train=1;
         imdb=[];
         input2={feat_update2 feat_preupdate2 label_update};
         opts.train.gpus=1;
         bopts.useGpu = numel(opts.train.gpus) > 0 ;
         
         info = cnn_train_dag_update(net_online2, imdb, input2,getBatchWrapper(bopts), ...
             trainOpts, ...
             'train', train, ...
             opts.train) ;
         net_online2.move('gpu');  
     end
     feat_update3{cur}=featPCA{3}; %feat_update{1}=前一帧的feature feat_update{2}=现在帧的feature
     feat_preupdate3{cur}=featPCA_pre{3};
     if cur==num_update
         
         trainOpts.batchSize = 1 ;
         trainOpts.numSubBatches = 1 ;
         trainOpts.continue = true ;
         trainOpts.gpus = 1 ;
         trainOpts.prefetch = true ;
         
         trainOpts.expDir = 'exp/update/' ;
         trainOpts.learningRate = 2e-9;
         trainOpts.weightDecay= 1;
         trainOpts.numEpochs = 2;
         
         train=1;
         imdb=[];
         input3={feat_update3 feat_preupdate3 label_update};
         opts.train.gpus=1;
         bopts.useGpu = numel(opts.train.gpus) > 0 ;
         
         info = cnn_train_dag_update(net_online3, imdb, input3,getBatchWrapper(bopts), ...
             trainOpts, ...
             'train', train, ...
             opts.train) ;
         net_online3.move('gpu');  
         cur=1;        
       else 
           cur=cur+1;            
     end

end

end

function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,false,opts,'prefetch',nargout==0) ;
end
