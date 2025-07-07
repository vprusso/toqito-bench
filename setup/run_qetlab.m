addpath('setup');
addpath(genpath('QETLAB/QETLAB-0.9'));
import matlab.perftest.TimeExperiment

% Ensure all parameters are correct numeric types

numSamples = 5
numWarmups = 10

exp = TimeExperiment.withFixedSampleSize(numSamples,'NumWarmups',numWarmups);

% Run the benchmarks
results = run(exp, testsuite('PartialTraceBenchmarks'));
T = sampleSummary(results);

% results  = runperf("PartialTraceBenchmarks");
%T        = sampleSummary(results);

ts       = string(datetime('now','Format','yyyy_MM_dd__HH_mm_ss'));
csvFile  = "detailed_" + ts + ".csv";

writetable(T,csvFile);

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

    if  startsWith(raw,"i")                 % scalar  i4  -> [4]
        newVal = extractAfter(raw,1);
        T.Name(k) = base + "[" + newVal + "]";

    elseif startsWith(raw,"l")              % list    l2_2 -> [[2, 2]]
        items  = strrep(extractAfter(raw,1),"_",", ");
        T.Name(k) = base + "[[" + items + "]]";

    else                                    % anything else -> wrap raw
        T.Name(k) = base + "[" + raw + "]";
    end
end

writetable(T,csvFile,"WriteMode","overwrite");
