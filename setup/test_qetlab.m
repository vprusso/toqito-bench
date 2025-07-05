% test_qetlab.m
% Demo QETLAB + CVX test for CI

disp('--- QETLAB CI Test Script ---');

% Add QETLAB and CVX to the path
addpath(genpath('QETLAB/QETLAB-0.9'));

classdef PartialTraceBenchmarks < matlab.perftest.TestCase
    properties (TestParameter)
         matrix_size = struct( ...
            'size_4', 4, ...
            'size_16', 16, ...
            'size_64', 64, ...
            'size_256', 256 ...
        );
    end
    
    methods (Test)
        function test_bench__partial_trace__vary__input_mat(testCase, matrix_size)
            input_mat = randn(matrix_size, matrix_size) + 1i*randn(matrix_size, matrix_size)
            sys = 2;
            dim = [];
            testCase.startMeasuring();
            result = PartialTrace(input_mat, sys, dim)
            testCase.stopMeasuring();
            testCase.verifyLessThanOrEqual(size(result,1), matrix_size);
        end
    end
end

disp('--- End of QETLAB CI Test Script ---');
