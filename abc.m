addpath("setup");
addpath(genpath("QETLAB/QETLAB-0.9"));

results  = runperf("PartialTraceBenchmarks");
T        = sampleSummary(results);
csvFile  = "PartialTraceBenchmarks_results.csv";
writetable(T,csvFile);                       % raw file with encoded names

% ---------------------------------------------------------------
% 2) Post-process the Name column
T = readtable(csvFile,"TextType","string");  % keep strings, not chars
dropPrefixExpr = "^[^/]+/";                 % everything before first “/”

for k = 1:height(T)
    disp("T.Name(" + k + ") = " + string(T.Name(k))); 
    disp(class(T.Name))
    s = regexprep(T.Name(k), dropPrefixExpr, "");  % remove class prefix
    
    % Extract pieces: base, param, raw value
    m = regexp(s, "^(?<base>[^()]+)\((?<param>[^=]+)=(?<val>[^)]+)\)$", ...
               "names");
    if isempty(m)
        continue                               % leave unchanged if it fails
    end
    
    base = string(m.base);
    raw = string(m.val);
    if startsWith(raw,"i")          % scalar  i4  -> [4]
        newVal = extractAfter(raw,1);
        T.Name(k) = base + "[" + newVal + "]";

    elseif startsWith(raw,"l")      % list    l2_2 -> [[2, 2]]
        items  = strrep(extractAfter(raw,1), "_", ", ");
        T.Name(k) = base + "[[" + items + "]]";

    else                            % anything else -> wrap raw
        T.Name(k) = base + "[" + raw + "]";
    end
end

writetable(T, csvFile, "WriteMode", "overwrite");  % overwrite with cleaned names
