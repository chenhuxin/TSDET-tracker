function info = Demo()

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', 1)};

run ('G:\TSDET\external\matconvnet\matlab\vl_setupnn') ;
addpath ('G:\TSDET\external\matconvnet\examples');

opts.expDir = 'exp/' ;
opts.dataDir = 'exp/data/' ;
opts.modelType = 'tracking';
opts.sourceModelPath = 'exp/models/' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = false ;

display=1;

g=gpuDevice(1);
clear g;     

show_plot=true;
test_seq='skiing';
title=test_seq;
[config]=config_list(test_seq);%结构体变量包括 imgList:1*81cell gt:81*4double nFrames:81 name:skiing
ground_truth=config.gt;

result=my_tracking(opts, varargin, config, display);
precisions=precision_plot(result,ground_truth, title, show_plot);
end
       



