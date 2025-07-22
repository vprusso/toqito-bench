classdef PartialTraceBenchmarks < matlab.perftest.TestCase

    properties (TestParameter)
        matrix_size = struct( ...
            'i4', 4, ...
            'i16', 16, ...
            'i64', 64, ...
            'i256', 256 ...
        );
        sys = struct( ...
            'l0', 1, ...
            'l1', 2, ...
            'l1_2', [1, 2], ...
            'l1_3', [1, 3] ...
        );
        dim = struct( ...
            'None', [], ...
            'l2_2', [2, 2], ...
            'l2_2_2_2', [2, 2, 2, 2], ...
            'l3_3', [3, 3], ...
            'l4_4', [4, 4] ...
        );
    end
   

    methods (TestClassSetup)
        function addQETLABToPath(~)
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
            matrix_size = 16;
            input_mat = randn(matrix_size, matrix_size) + 1i * randn(matrix_size, matrix_size);

            if isequal(sys, [1, 2])
                dim = [4, 4];
            elseif isequal(sys, [1, 3])
                dim = [2, 2, 2, 2];
            else
                dim = [4,4];
            end

            testCase.startMeasuring();
            result = PartialTrace(input_mat, sys, dim);
            testCase.stopMeasuring();
            testCase.verifyNotEmpty(result);
        end

        function test_bench__partial_trace__vary__dim(testCase, dim)
            if isempty(dim)
                matrix_size = 16;
                dim = [4,4];
            else
                matrix_size = prod(dim);
            end

            input_mat = randn(matrix_size, matrix_size) + 1i * randn(matrix_size, matrix_size);
            sys = [];

            testCase.startMeasuring();
            result = PartialTrace(input_mat, sys, dim);
            testCase.stopMeasuring();
            testCase.verifyNotEmpty(result);
        end
    end
end

