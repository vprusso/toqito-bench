
% Demo MATLAB script for CI

disp('--- MATLAB CI Demo Script ---');
disp(['MATLAB Version: ', version]);
disp(['Current folder: ', pwd]);
disp(['Current date/time: ', datestr(now)]);
[status, hostname] = system('hostname');
if status == 0
    disp(['Running on host: ', strtrim(hostname)]);
else
    disp('Could not retrieve hostname.');
end

randNums = rand(1,3);
disp('Three random numbers:');
disp(randNums);

a = 5;
b = 3;
sum_ab = a + b;
disp(['Sum of ', num2str(a), ' and ', num2str(b), ' is: ', num2str(sum_ab)]);

if sum_ab == 8
    disp('Test passed: sum is correct.');
else
    disp('Test failed: sum is incorrect.');
end

disp('--- End of MATLAB CI Demo Script ---');
