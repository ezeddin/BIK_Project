function result = CheckNetDerivative(net,x,opts)
%x must be a vector
defaultOpts = struct(...
        'delta',.0001,...
        'plot',false);
    
if nargin <= 2
    opts = defaultOpts;
else
    if ~isfield(opts,'delta')
        opts.delta = defaultOpts.delta;
    end
    if ~isfield(opts,'plot')
        opts.plot = false;
    end
end
if nargin <= 1 || isempty(x)
    x = randn(10,1,'single');
end
if nargin <= 0 || isempty(net)
    net = resUnit.getTestNetwork(size(x));
end
dzdy = randn(size(x), 'single');
res = vl_simplenn(net,x,dzdy);
y = res(end).x;
dzdx = res(1).dzdx;
dzdx_num = zeros(size(dzdx));
last = @(x) x(end);
f = @(x) getfield(last(vl_simplenn(net,x)),'x');
for i = 1:length(x)
    d = zeros(length(y),1);
    d(i) = opts.delta;
    for j = 1:length(y)
        dy_jdx_i = ((f(x+d) - f(x-d))/(2*opts.delta));
        dy_jdx_i = dy_jdx_i(j);
        dzdx_num(i) = dzdx_num(i) + dzdy(j) * dy_jdx_i;
    end
end
dzdx_num = dzdx_num./1;
if opts.plot
    axis equal
    scatter(dzdx_num,dzdx);
    xlabel('numeric derivative');
    ylabel('calculated derivative');
end
result.dzdx = dzdx;
result.dzdx_num = dzdx_num;
end 