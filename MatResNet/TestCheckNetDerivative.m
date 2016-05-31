%Prints test plots for the CheckNetDerivative function.

workingNet.layers = {};
workingNet.layers{end + 1} =  struct(...
                'name', sprintf('relu %d',1),...
                'type', 'relu',...
                'leak',.2);
workingNet = vl_simplenn_tidy(workingNet);
result = CheckNetDerivative(workingNet,randn(100,1,'single'),struct('plot',false,'delta',.000001));
figure;
axis equal;
scatter(result.dzdx_num,result.dzdx);
xlabel('numeric derivative');
ylabel('calculated derivative');
title('Should be correct');

brokenNet.layers = {};
brokenNet.layers{end + 1} = struct(...
    'name', 'broken layer',...
    'type', 'custom',...
    'forward',@(layer,res,res_) setfield(res_,'x',res.x),...
    'backward',@(layer,res,res_) setfield(res,'dzdx',zeros(size(res_.x))));
brokenNet = vl_simplenn_tidy(brokenNet);
result = CheckNetDerivative(brokenNet,randn(100,1,'single'),struct('plot',false,'delta',.000001));
figure;
axis equal;
scatter(result.dzdx_num,result.dzdx);
xlabel('numeric derivative');
ylabel('calculated derivative');
title('Should be incorrect');