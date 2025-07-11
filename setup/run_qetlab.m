addpath('setup');
addpath(genpath('QETLAB/QETLAB-0.9'));
import matlab.perftest.TimeExperiment


numSamples = 5;
numWarmups = 100;

% Create a time experiment with a fixed number of samples.
exp = TimeExperiment.withFixedSampleSize(numSamples,'NumWarmups',numWarmups);

results = run(exp, testsuite('PartialTraceBenchmarks'));
T = sampleSummary(results);

% Generate a timestamped filename for the results CSV.
ts       = string(datetime('now','Format','yyyy_MM_dd__HH_mm_ss'));
csvFile  = "detailed_" + ts + ".csv";

% Write the initial results to the CSV file.
writetable(T,csvFile);
% Read the results back to process the test names for clarity.
T = readtable(csvFile,"TextType","string"); % keep strings, not chars
dropPrefixExpr = "^[^/]+/";                 % everything before first “/”

for k = 1:height(T)
    s = regexprep(T.Name(k), dropPrefixExpr, "");

    m = regexp(s, ...
        "^(?<base>[^()]+)\((?<param>[^=]+)=(?<val>[^)]+)\)$", ...
        "names");
    if isempty(m),  continue,  end

    base = string(m.base);
    raw  = string(m.val);

    % Format the test name based on the parameter type.
    if  startsWith(raw,"i") % Scalar integer; i4  -> [4]
        newVal = extractAfter(raw,1);
        T.Name(k) = base + "[" + newVal + "]";

    elseif startsWith(raw,"l")% list  l2_2 -> [[2, 2]]
        items  = strrep(extractAfter(raw,1),"_",", ");
        T.Name(k) = base + "[[" + items + "]]";
    else
        T.Name(k) = base + "[" + raw + "]";
    end
end

writetable(T,csvFile,"WriteMode","overwrite");
