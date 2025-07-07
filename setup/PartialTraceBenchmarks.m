classdef PartialTraceBenchmarks < matlab.perftest.TestCase
    properties (TestParameter)
        matrix_size = struct( ...
            '[4]', 4, ...
            '[16]', 16, ...
            '[64]', 64, ...
            '[256]', 256, ...
        );

        sys = struct(...
            '[0]', 1, ...
            '[1]', 2, ...
            '[0, 1]', [1, 2], ...
            '[0, 2]', [1, 3], ...
        );

        dim = struct( ...
            'None', [], ...
            '[2, 2]', [2, 2], ...
            '[2, 2, 2, 2]', [2, 2, 2, 2], ...
            '[3, 3]', [3, 3], ...
            '[4, 4]', [4, 4], ...
        )
    end

    methods (TestClassSetup)
        function addQETLABToPath(testCase)
            addpath(genpath('QETLAB/QETLAB-0.9'));
        end
    end

    methods (Test)
        function test_bench__partial_trace__vary__input_mat(testCase, matrix_size)
            input_mat = randn(matrix_size, matrix_size) + 1i*randn(matrix_size, matrix_size);
            sys = 2;
            d = sqrt(matrix_size);
            dim = [d, d];
            testCase.startMeasuring();
            result = PartialTrace(input_mat, sys, dim);
            testCase.stopMeasuring();
            testCase.verifyLessThanOrEqual(size(result,1), matrix_size);
        end

        function test_bench__partial_trace__vary__sys(testCase, sys)
            % Fixed matrix size of 16 x 16
            matrix_size = 16;
            input_mat= randn(matrix_size, matrix_size) + 1i * randn(matrix_size, matrix_size)

            % Adjust dim based on sys
            if isequal(sys, [1, 2])
                dim = [4, 4]; % Two subsystems of dim 4
            elseif isequal(sys, [1, 3])
                dim = [2, 2, 2, 2]; % Four qubits
            else
                dim = [] % Use default dim handling
            end

            testCase.startMeasuring();
            result = PartialTrace(input_mat, sys, dim);
            testCase.stopMeasuring();

            testCase.verifyNotEmpty(result);
        
        function test_bench__partial_trace__vary__dim(testCase, dim)
            % Set matrix size based on dim
            if isempty(dim)
                matrix_size = 16;
            else
                matrix_size = prod(dim);
            end

            input_mat = randn(matrix_size, matrix_size) + 1i * randn(matrix_size, matrix_size);
            sys = [];  % Use default sys handling

            testCase.startMeasuring();
            result = PartialTrace(input_mat, sys, dim);
            testCase.stopMeasuring();

            testCase.verifyNotEmpty(result);

    end
end
