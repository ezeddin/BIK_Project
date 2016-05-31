classdef resUnit
    methods(Static)
        function val = new(i,inputDim,outputDim)
            %
            %
            p = ptr();
            split = struct(...
                'name', sprintf('resBegin %d',i),...
                'type', 'custom',...
                'forward',@resUnit.splitForward,...
                'backward',@resUnit.splitBackward,...
                'resEnd',p);
            res = struct(...
                'name', sprintf('resEnd %d',i),...
                'type', 'custom',...
                'forward',@resUnit.resForward,...
                'backward',@resUnit.resBackward,...
                'resBegin',p);
            if nargin >= 3
                W = xavier(1,1,inputDim,outputDim);
                split.F = W{1};
                split.B = W{2};
            end
            val = {split, res};
        end
        function testGradient()
            x0 = randn(size(x), 'single') ;

            %forward = @(; backward = @l2LossBackward;

            % Uncomment the followung line to test your L1 loss implementation
            % forward = @l1LossForward; backward = @l1LossBackward;

            y = forward(x, x0);

            p = randn(size(y), 'single');
            dx = backward(x, x0, p);

            % Check the derivative numerically
            figure(23) ; clf('reset') ;
            set(gcf, 'name', 'Part 2.3: custom loss layer') ;
            func = @(x) proj(p, forward(x, x0)) ;
            checkDerivativeNumerically(func, x, dx);
        end
        
        function testNet = getTestNetwork(inputSizes)
            if nargin <= 0
                inputSizes = [10 10 1 2];
            end
            testNet.meta.inputSize = inputSizes;
            testNet.layers = {};
            resU = resUnit.new(1);
            testNet.layers{end + 1} = resU{1};
            testNet.layers{end + 1} = struct(...
                'name', sprintf('relu %d',1),...
                'type', 'relu',...
                'leak',.2);
            testNet.layers{end + 1} = resU{2};
            testNet = vl_simplenn_tidy(testNet);
        end
    end
    methods(Access = private, Static)
        function res_ = splitForward(layer,res,res_)
            if isfield(layer,'F')
            	layer.resEnd.val.x = vl_nnconv(res.x,layer.F,layer.B);
            else
                layer.resEnd.val.x = res.x;
            end
            res_.x = res.x;
        end

        function res = splitBackward(layer,res,res_)
            if isfield(layer,'F')
                [dzdxp,dzdF,dzdB] = vl_nnconv(res_.x,layer.F,layer.B,layer.resEnd.val.dzdx);
                layer.F = layer.F - dzdF .* layer.learningrate(1);
                layer.B = layer.B - dzdB .* layer.learningrate(2);
            else
                dzdxp = layer.resEnd.val.dzdx;
            end            
            res.dzdx = (res_.dzdx + dzdxp);
        end

        function res_ = resForward(layer,res,res_)
            res_.x = res.x + layer.resBegin.val.x;
        end

        function res = resBackward(layer,res,res_)
            res.dzdx = res_.dzdx;
            layer.resBegin.val.dzdx = res_.dzdx;
        end
        
    end
end