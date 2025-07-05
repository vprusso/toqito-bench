% test_qetlab.m
% Demo QETLAB + CVX test for CI

disp('--- QETLAB CI Test Script ---');

% Add QETLAB and CVX to the path
addpath(genpath('QETLAB/QETLAB-0.9'));

rho_rand = RandomDensityMatrix(9);
is_ppt = IsPPT(rho_rand);
disp(['Random 9x9 state is PPT: ', num2str(is_ppt)]);


disp('--- End of QETLAB CI Test Script ---');
