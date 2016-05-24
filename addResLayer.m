function net = addResLayer(net)
%ADDCUSTOMLOSSLAYER Add a custom loss layer to a network
%   NET = ADDCUSTOMLOSSLAYER(NET, FWDFUN, BWDFUN) adds a custom loss
%   layer to the network NET using FWDFUN for forward pass and BWDFUN for
%   a backward pass.

layer.name = sprintf('res %d',length(net.layers));
layer.type = 'custom' ;
layer.forward = @forward;
layer.backward = @backward;
net.layers{end+1} = layer ;

  function res_ =  forward(layer, res, res_)
    %BN
    vl_nnbnorm();
    %Relu

    %Conv

    %BN

    %Relu

    %Conv

      res_.x = fwfun(res.x, layer.class) ;
  end

  function res = backward(layer, res, res_)
    res.dzdx = bwfun(res.x, layer.class, res_.dzdx) ;
  end
end


