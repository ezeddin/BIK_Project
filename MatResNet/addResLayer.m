function net = addResLayer(net)
%ADDCUSTOMLOSSLAYER Add a custom loss layer to a network
%   NET = ADDCUSTOMLOSSLAYER(NET, FWDFUN, BWDFUN) adds a custom loss
%   layer to the network NET using FWDFUN for forward pass and BWDFUN for
%   a backward pass.

layer.name = sprintf('res %d',length(net.layers));
layer.type = 'custom' ;
layer.forward = @forward;
layer.backward = @backward;

%TODO: Check how to propperly initialize batch normalisation layers;
%TODO: Determine size of input.
convInit = xavier();
group = struct(...
    'bn',struct(...
            'G',1,...
            'B',1,...
            'x',0),...
    'relu',struct(...
            'x',0),...
    'conv',struct(...
            'F',convInit{1},...
            'B',convInit{2},...
            'x',0));
        
layer.group = [group,group];
net.layers{end+1} = layer ;
    function res_ =  forward(layer, res, res_)
        %BN
        layer.group(1).bn.x = vl_nnbnorm(res.x,layer.group(1).bn.G,layer.group(1).bn.B);
        %Relu
        layer.group(1).relu.x = vl_nnrelu(layer.group(1).bn.x);
        %Conv
        layer.group(1).conv.x = vl_nnconv(...
            layer.group(1).relu.x,layer.group(1).conv.F,layer.group(1).conv.B);
        %BN
        layer.group(2).bn.x = vl_nnbnorm(...
            layer.group(1).conv.x,layer.group(2).bn.G,layer.group(2).bn.B);
        %Relu
        layer.group(2).relu.x = vl_nnrelu(layer.group(2).bn.x);
        %Conv
        layer.group(2).conv.x = vl_nnconv(...
            layer.group(2).relu.x,layer.group(2).conv.F,layer.group(2).conv.B);
        %Res
        res_.x = res.x + layer.group(2).conv.x;
    end

  function res = backward(layer, res, res_)
    res.dzdx = bwfun(res.x, layer.class, res_.dzdx) ;
  end
end


