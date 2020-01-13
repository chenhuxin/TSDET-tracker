 function results = run_TSDET(seq,ground_truth, res_path, bSaveImage)
    varargin=cell(1,2);

    varargin(1,1)={'train'};
    varargin(1,2)={struct('gpus', 1)};

    run ('G:\OTB\tracker_benchmark_v1.0\trackers\TSEHCFT\external\matconvnet\matlab\vl_setupnn') ;
    addpath ('G:\OTB\tracker_benchmark_v1.0\trackers\TSEHCFT\external\matconvnet\examples') ;

    opts.expDir = 'exp/' ;
    opts.dataDir = 'exp/data/' ;
    opts.modelType = 'tracking' ;
    opts.sourceModelPath = 'exp/models/' ;
    [opts, varargin] = vl_argparse(opts, varargin) ;

    % experiment setup
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
    opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
    opts.vocEdition = '11' ;
    opts.vocAdditionalSegmentations = false ;

    %global resize;
    display=0;

    %g=gpuDevice(1);
    %clear g;                             
    [config]=config_list(seq,ground_truth);%结构体变量包括 imgList:1*81cell gt:81*4double nFrames:81 name:skiing
    

    rects=my_tracking(opts, varargin, config, display);
    if bSaveImage
        imwrite(frame2im(getframe(gcf)),[res_path num2str(frame) '.jpg']);
    end

    results.type = 'rect';
    results.res = rects;%each row is a rectangle
  end

