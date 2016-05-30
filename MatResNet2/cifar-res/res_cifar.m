setup() ;
% setup('useGpu', true); % Uncomment to initialise with a GPU support
%% Part 3.1: Prepare the data

% Load a database of blurred images to train from
opts.dataDir = 'data';
opts.whitenData = true ;
opts.contrastNormalization = true ;

if ~exist('imdb','var')
    imdb = getCifarImdb(opts);
end

net = res_cifar_init([32 32 3 1],3);
net.meta.classes.name  = imdb.meta.classes(:)' ;

%% Part 3.3: learn the model

% Add a loss (using a custom layer)
lossLayer = struct(...
                    'name','loss layer',...
                    'type','loss',...
                    'class',[]);

% Make sure that the loss layer is not added multiple times
if strcmp(net.layers{end}.name, lossLayer.name)
  net.layers{end} = lossLayer ;
else
  net.layers{end+1} = lossLayer ;
end
net = vl_simplenn_tidy(net) ;

% Extra: uncomment the following line to use your implementation
% of the L1 loss
%net = addCustomLossLayer(net, @l1LossForward, @l1LossBackward) ;

% Train
trainOpts.expDir = 'data/cifar-epochs' ;
trainOpts.gpus = [] ;
% Uncomment for GPU training:
%trainOpts.expDir = 'data/text-small-gpu' ;
%trainOpts.gpus = [1] ;
trainOpts.batchSize = 100 ;
trainOpts.learningRate = 0.0001 ;
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 15 ;
trainOpts.errorFunction = 'none' ;


net = cnn_train(net, imdb, @getCifarBatch, trainOpts) ;

% Deploy: remove loss
net.layers(end) = [] ;

%% Part 3.4: evaluate the model

train = find(imdb.images.set == 1) ;
val = find(imdb.images.set == 3) ;

i = round(rand(1)*60000);
im_ = imdb.images.data(:,:,:,i) ;

% % Evaluate network on an image
res = vl_simplenn(net, im_) ;

scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im_) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.name{best}, best, bestScore)) ;
