function net = res_mnist_init_dagnn(varargin)
opts.architecture = 'ResNet';
opts = vl_argparse(opts, varargin);

net = dagnn.DagNN();
net.meta.inputSize = [28 28 1]; 
net.meta.trainOpts.weightDecay = 0.0001 ; 
net.meta.trainOpts.momentum = 0.9; 
net.meta.trainOpts.batchSize = 100 ; 
net.meta.trainOpts.learningRate = ...
    [0.01*ones(1,20) 0.001*ones(1,20) 0.0001*ones(1,10)];
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate);

outputName = add_Res_Group(net,struct(...
    'inputName','input',...
    'groupIndex',1,...
    'inputChannels',1,...
    'outputChannels',16,...
    'resUnits',4));
outputName = add_Res_Group(net,struct(...
    'inputName',outputName,...
    'groupIndex',2,...
    'inputChannels',16,...
    'outputChannels',32,...
    'resUnits',4));

outputName = add_Res_Group(net,struct(...
    'inputName',outputName,...
    'groupIndex',3,...
    'inputChannels',32,...
    'outputChannels',64,...
    'resUnits',4));

%Fully cunnected layer
add_conv_layer(net,struct(...
    'name','final_fc_layer',...
    'inputName', outputName,...
    'outputName','x_L',...
    'size',[4,4,64,10],...
    'stride',1,...
    'pad',0));

%Loss
net.addLayer('softmax', dagnn.SoftMax(), 'x_L', 'x_L_1');   
net.addLayer('loss', dagnn.Loss('loss', 'log'), {'x_L_1', 'label'}, 'x_L_2'); 
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'x_L_2','label'}, 'objective') ; 

 
net.initParams(); 



end



function outputName = add_Res_Group(net,opts)
%Opts must contain
%   inputName :: string
%   groupIndex :: int
%   inputChannels :: int
%   outputChannels :: int
%   resUnits :: int

    outputName = add_res_unit(net,struct(...
        'inputName',opts.inputName,...
        'groupIndex',opts.groupIndex,...
        'resUnitIndex',1,...
        'inputChannels',opts.inputChannels,...
        'outputChannels',opts.outputChannels,...
        'stride',2));
    for i = 2:opts.resUnits
    outputName = add_res_unit(net,struct(...
        'inputName',outputName,...
        'groupIndex',opts.groupIndex,...
        'resUnitIndex',i,...
        'inputChannels',opts.outputChannels,...
        'outputChannels',opts.outputChannels,...
        'stride',1));
    end
end

function outputName = add_res_unit(net,opts)
%Opts must contains
%   inputName :: string
%   groupIndex :: int
%   resUnitIndex :: int
%   inputChannels :: int
%   outputChannels :: int
%   stride :: int
    layer_index = [opts.groupIndex,opts.resUnitIndex,1];
    add_bn_layer(net,struct(...
        'name',get_layer_name('BN',layer_index),...
        'inputName',opts.inputName,...
        'outputName',get_x_out_name(layer_index),...
        'channels',opts.inputChannels));
    
    prev_layer_index = layer_index;
    layer_index(3) = 2;
    add_relu_layer(net,struct(...
        'name',get_layer_name('relu',layer_index),...
        'inputName',get_x_out_name(prev_layer_index),...
        'outputName',get_x_out_name(layer_index)));
    
    prev_layer_index = layer_index;
    layer_index(3) = 3;
    add_conv_layer(net,struct(...
        'name',get_layer_name('conv',layer_index),...
        'inputName',get_x_out_name(prev_layer_index),...
        'outputName',get_x_out_name(layer_index),...
        'size',[3,3,opts.inputChannels,opts.outputChannels],...
        'pad',1,...
        'stride',opts.stride));
    
    prev_layer_index = layer_index;
    layer_index(3) = 4;
    add_bn_layer(net,struct(...
        'name',get_layer_name('BN',layer_index),...
        'inputName',get_x_out_name(prev_layer_index),...
        'outputName',get_x_out_name(layer_index),...
        'channels',opts.outputChannels));
    
    prev_layer_index = layer_index;
    layer_index(3) = 5;
    add_relu_layer(net,struct(...
        'name',get_layer_name('relu',layer_index),...
        'inputName',get_x_out_name(prev_layer_index),...
        'outputName',get_x_out_name(layer_index)));
    
    prev_layer_index = layer_index;
    layer_index(3) = 6;
    add_conv_layer(net,struct(...
        'name',get_layer_name('relu',layer_index),...
        'inputName',get_x_out_name(prev_layer_index),...
        'outputName',get_x_out_name(layer_index),...
        'size',[3,3,opts.outputChannels,opts.outputChannels],...
        'pad',1,...
        'stride',1));
    
    prev_layer_index = layer_index;
    layer_index = [opts.groupIndex,opts.resUnitIndex];
    outputName = get_x_out_name(layer_index);
    if opts.inputChannels == opts.outputChannels
        net.addLayer(get_layer_name('Add',layer_index),dagnn.Sum(),...
            {opts.inputName,get_x_out_name(prev_layer_index)},...
            outputName);
    else
        
        %TODO: add projection
        add_conv_layer(net,struct(...
            'name',sprintf('project_%d_%d',layer_index),...
            'inputName',opts.inputName,...
            'outputName',get_x_out_name(layer_index),...
            'size',[1,1,opts.inputChannels,opts.outputChannels],...
            'pad',0,...
            'stride',opts.stride));
        
        net.addLayer(sprintf('Add_%d_%d',layer_index),dagnn.Sum(),...
            {get_x_out_name(layer_index),get_x_out_name(prev_layer_index)} ,outputName);
    end 
    
end

function add_conv_layer(net,opts)
    %Opts must contain the fields
    %   name :: string
    %   inputName :: string
    %   outputName :: string
    %   size :: [int] , length 4.
    %   stride :: int
    %   pad :: int
    block = dagnn.Conv(...
        'size',opts.size,...
        'hasBias',true,...
        'stride',opts.stride,...
        'pad',opts.pad);
    parameter_names = {[opts.name '_f'] [opts.name '_b']};
    net.addLayer(opts.name,block,opts.inputName,opts.outputName,parameter_names);
    net.params(net.getParamIndex(parameter_names(2))).weightDecay = 0; % Why set this to 0?
end

function add_bn_layer(net,opts)
    %Opts must contain the fields
    %   name :: string
    %   inputName :: string
    %   outputName :: string
    %   channels :: int
    %Opts may contain the fields
    %   learningRate :: double
    block = dagnn.BatchNorm('numChannels',opts.channels);
    parameter_names = {[opts.name '_g'], [opts.name '_b'], [opts.name '_m']};
    net.addLayer(opts.name,block,opts.inputName,opts.outputName,parameter_names);
    parameter_indexes = net.getParamIndex(parameter_names);
    net.params(parameter_indexes(1)).weightDecay = 0;
    net.params(parameter_indexes(2)).weightDecay = 0;
    if ~isfield('learningRate',opts)
        lr = .1;
    else
        lr = opts.learningRate;
    end
    net.params(parameter_indexes(3)).learningRate = lr;
    net.params(parameter_indexes(3)).trainMethod = 'average';
end

function x_name = get_x_out_name(layer_index)
    switch length(layer_index)
        case 3 
            x_name = sprintf('x_%d_%d_%d',layer_index);
        case 2
            x_name = sprintf('x_%d_%d',layer_index);
    end
end

function layer_name = get_layer_name(prefix,layer_index)
    switch length(layer_index)
        case 3 
            layer_name = sprintf('%s_%d_%d_%d',prefix,layer_index);
        case 2
            layer_name = sprintf('%s_%d_%d',prefix,layer_index);
    end
end

function add_relu_layer(net,opts)
    %Opts must contain
    %   name :: string
    %   inputName :: string
    %   outputName :: string
    %Opts may contain the fields
    %   leak
    if isfield(opts,'leak')
        leak = opts.leak;
    else
        leak = 0;
    end
    block = dagnn.ReLU('leak',leak);
    net.addLayer(opts.name,block,opts.inputName,opts.outputName);
end