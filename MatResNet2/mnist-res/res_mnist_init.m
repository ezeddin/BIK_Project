function net = res_mnist_init(varargin)
% RES_MNIST_LENET Initialize a CNN similar for MNIST
opts.networkType = 'ResNet' ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

net.layers = {} ;

net = addResGroup(net,'16',[28,28,1],[14,14,16],3,opts);
net = addResGroup(net,'32',[14,14,16],[7,7,32],3,opts);
net = addResGroup(net,'64',[7,7,32],[4,4,64],3,opts);

net.layers{end+1} = struct( ...
                    'name',  'fc',...
                    'type', 'conv',...
                    'weights', {xavier(4,4,64,10)},...
                    'pad', 0);

net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [27 27 1] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 25 ;
net.meta.trainOpts.batchSize = 100 ;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

switch lower(opts.networkType)
  case 'resnet'
  case 'plain'
  otherwise
    assert(false) ;
end

end
function net = addResGroup(net, name, inputDims, outputDims, n,opts)
    stride = 2;

    % ------ intial dimentionalty adjustment -------
    resUnitName = sprintf('%s,1', name);
    inputChannels = inputDims(3);
    outputChannels = outputDims(3);
    
    if strcmpi(opts.networkType,'resnet')
        resLayers = resUnit.new(resUnitName,inputChannels,outputChannels,stride);
        net.layers{end+1} = resLayers{1};
    end
    net.layers{end+1} = newBnorm(sprintf('BN 1 - %s',resUnitName),inputChannels);
    net.layers{end+1} = struct(...
                'name', sprintf('relu 1 - %s',resUnitName),...
                'type', 'relu') ;
    net.layers{end+1} = struct( ...
                'name',  sprintf('conv 1 - %s',resUnitName),...
                'type', 'conv',...
                'learningrate', [1 1 0.1],...
                'weights', {xavier(3,3,inputChannels,outputChannels)},...
                'stride',stride,...
                'pad', 1);
    
    net.layers{end+1} = newBnorm(sprintf('BN 2 - %s',resUnitName),outputChannels);
    net.layers{end+1} = struct(...
                'name', sprintf('relu 2 - %s',resUnitName),...
                'type', 'relu') ;
    net.layers{end+1} = struct( ...
                'name', sprintf('conv 2 - %s',resUnitName),...
                'type', 'conv',...
                'learningrate', [1 1 0.1],...
                'weights', {xavier(3,3,outputChannels,outputChannels)},...
                'pad', 1);
    
    if strcmpi(opts.networkType,'resnet')
        net.layers{end+1} = resLayers{2};
    end
    % ----------------------------------------------
    
    for i = 2:n
        resUnitName = sprintf('%s,%d', name,i);
        
        if strcmpi(opts.networkType,'resnet')
            resLayers = resUnit.new(resUnitName);
            net.layers{end+1} = resLayers{1};
        end
        net.layers{end+1} = newBnorm(sprintf('BN 1 - %s',resUnitName),outputChannels);
        net.layers{end+1} = struct(...
                    'name', sprintf('relu 1 - %s',resUnitName),...
                    'type', 'relu') ;
        net.layers{end+1} = struct( ...
                    'name',  sprintf('conv 1 - %s',resUnitName),...
                    'type', 'conv',...
                    'learningrate', [1 1 0.1],...
                    'weights', {xavier(3,3,outputChannels,outputChannels)},...
                    'pad', 1);

        net.layers{end+1} = newBnorm(sprintf('BN 2 - %s',resUnitName),outputChannels);
        net.layers{end+1} = struct(...
                    'name', sprintf('relu 2 - %s',resUnitName),...
                    'type', 'relu') ;
        net.layers{end+1} = struct( ...
                    'name', sprintf('conv 2 - %s',resUnitName),...
                    'type', 'conv',...
                    'learningrate', [1 1 0.1],...
                    'weights', {xavier(3,3,outputChannels,outputChannels)},...
                    'pad', 1);
        if strcmpi(opts.networkType,'resnet')
            net.layers{end+1} = resLayers{2};
        end
    end
end
% --------------------------------------------------------------------
function layer = newBnorm(name,ndim)
    layer = struct('name', name,...
                   'type', 'bnorm', ...
                   'learningRate', [1 1 0.05], ...
                   'weightDecay', [0 0]) ;
    layer.weights = {ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')};
end

