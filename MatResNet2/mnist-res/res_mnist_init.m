function net = res_mnist_init(varargin)
% RES_MNIST_LENET Initialize a CNN similar for MNIST
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=0.0001 ;
net.layers = {} ;

resLayers = resUnit.new(1);
resBegin1 = resLayers{1};    
resEnd1 = resLayers{2};

resLayers = resUnit.new(2);
resBegin2 = resLayers{1};    
resEnd2 = resLayers{2};

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
%net.layers{end+1} = resBegin1;
net.layers{end+1} = struct('type', 'relu');
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,20,20, 'single'),zeros(1,20,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
%net.layers{end + 1} = resEnd1;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(4,4,20,50, 'single'),zeros(1,50,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
                       
%net.layers{end+1} = resBegin2;
net.layers{end+1} = struct('type', 'relu');
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,500,500, 'single'),  zeros(1,500,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
%net.layers{end + 1} = resEnd2;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,500,10, 'single'), zeros(1,10,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% % optionally switch to batch normalization
% if opts.batchNormalization
%   net = insertBnorm(net, 5) ;
%   net = insertBnorm(net, 12) ;
% end

% Meta parameters
net.meta.inputSize = [27 27 1] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 10 ;
net.meta.trainOpts.batchSize = 100 ;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
             {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1 0.05], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
