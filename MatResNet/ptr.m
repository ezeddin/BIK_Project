classdef ptr < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        val
    end
    
    methods
        function obj = ptr(val)
            if nargin ~= 0
                obj.val = val;
            end
        end
    end
end

