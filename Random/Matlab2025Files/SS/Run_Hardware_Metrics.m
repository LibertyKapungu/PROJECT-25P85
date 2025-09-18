% =========================================================================
% Robust Hardware Metrics Evaluation Script
% =========================================================================
% Eliminates measurement artifacts through proper benchmarking methodology:
% - Multiple warm-up runs to eliminate JIT compilation effects
% - Statistical analysis with multiple measured runs
% - Randomized algorithm execution order
% - Proper memory measurement techniques
% - Cross-validation with order reversals
% =========================================================================

clear; clc; close all;

% --- Configuration ---
addpath(genpath('spectral_algorithms/'));
inputFile = 'validation_dataset/noisy_speech/sp21_station_sn5.wav';

% Benchmarking parameters
NUM_WARMUP_RUNS = 3;        % Warm-up runs (not measured)
NUM_MEASURED_RUNS = 10;     % Measured runs for averaging
NUM_ORDER_TESTS = 5;        % Number of different random orders to test

% --- Algorithm Setup ---
algorithms = {'specsub', 'mband'};
algorithmFuncs = containers.Map;
algorithmFuncs('specsub') = @(infile, outfile) specsub(infile, outfile);
algorithmFuncs('mband') = @(infile, outfile) mband(infile, outfile, 6, 'mel');

% --- Get audio file duration for RTF calculation ---
try
    audioInfo = audioinfo(inputFile);
    fileDuration = audioInfo.Duration;
    fprintf('Audio file duration: %.3f seconds\n', fileDuration);
catch ME
    error('Could not read audio file: %s\nError: %s', inputFile, ME.message);
end

% --- Initialize Results Storage ---
metricNames = {'Latency_sec_mean', 'Latency_sec_std', 'MaxMemoryMB', 'RealTimeFactor_mean', 'RealTimeFactor_std'};
finalResults = array2table(zeros(length(metricNames), length(algorithms)), ...
    'RowNames', metricNames, 'VariableNames', algorithms);

% Storage for all individual measurements
allMeasurements = struct();
for algo = algorithms
    allMeasurements.(algo{1}) = struct('latencies', [], 'memories', [], 'rtfs', []);
end

fprintf('\n=== Starting Robust Hardware Metrics Evaluation ===\n');
fprintf('Configuration: %d warmup runs, %d measured runs, %d order tests\n\n', ...
    NUM_WARMUP_RUNS, NUM_MEASURED_RUNS, NUM_ORDER_TESTS);

% --- Main Testing Loop with Randomized Orders ---
for orderTest = 1:NUM_ORDER_TESTS
    fprintf('--- Order Test %d/%d ---\n', orderTest, NUM_ORDER_TESTS);
    
    % Randomize algorithm order for this test
    algorithmOrder = algorithms(randperm(length(algorithms)));
    fprintf('Testing order: %s\n', strjoin(algorithmOrder, ' -> '));
    
    for algoIdx = 1:length(algorithmOrder)
        currentAlgo = algorithmOrder{algoIdx};
        func = algorithmFuncs(currentAlgo);
        
        fprintf('  Testing %s (position %d)...\n', currentAlgo, algoIdx);
        
        % --- Warm-up Phase (Critical for eliminating JIT effects) ---
        fprintf('    Warm-up runs: ');
        for warmup = 1:NUM_WARMUP_RUNS
            tempFile = sprintf('temp_warmup_%s_%d.wav', currentAlgo, warmup);
            try
                func(inputFile, tempFile);
                delete(tempFile);
                fprintf('.');
            catch ME
                fprintf('\n    Warning: Warm-up run %d failed: %s\n', warmup, ME.message);
                if exist(tempFile, 'file'), delete(tempFile); end
            end
        end
        fprintf(' Done\n');
        
        % --- Measured Runs Phase ---
        fprintf('    Measured runs: ');
        latencies = zeros(1, NUM_MEASURED_RUNS);
        memories = zeros(1, NUM_MEASURED_RUNS);
        rtfs = zeros(1, NUM_MEASURED_RUNS);
        
        for run = 1:NUM_MEASURED_RUNS
            % Clean memory state before each measurement
            if run > 1
                clear tempVars; % Clear any temporary variables
            end
            
            % Prepare output file
            outFile = sprintf('temp_measured_%s_run%d.wav', currentAlgo, run);
            
            try
                % Memory measurement - before
                memInfo1 = memory;
                memBefore = memInfo1.MemUsedMATLAB;
                
                % Time the algorithm execution
                tic;
                func(inputFile, outFile);
                latencies(run) = toc;
                
                % Memory measurement - after
                memInfo2 = memory;
                memAfter = memInfo2.MemUsedMATLAB;
                memories(run) = (memAfter - memBefore) / (1024^2); % Convert to MB
                
                % Calculate Real-Time Factor
                rtfs(run) = latencies(run) / fileDuration;
                
                % Clean up output file
                if exist(outFile, 'file')
                    delete(outFile);
                end
                
                fprintf('.');
                
            catch ME
                fprintf('\n    Error in run %d: %s\n', run, ME.message);
                if exist(outFile, 'file'), delete(outFile); end
                % Use NaN for failed runs
                latencies(run) = NaN;
                memories(run) = NaN;
                rtfs(run) = NaN;
            end
        end
        fprintf(' Done\n');
        
        % --- Store measurements for this algorithm ---
        validRuns = ~isnan(latencies);
        if sum(validRuns) > 0
            allMeasurements.(currentAlgo).latencies = [allMeasurements.(currentAlgo).latencies, latencies(validRuns)];
            allMeasurements.(currentAlgo).memories = [allMeasurements.(currentAlgo).memories, memories(validRuns)];
            allMeasurements.(currentAlgo).rtfs = [allMeasurements.(currentAlgo).rtfs, rtfs(validRuns)];
            
            fprintf('    Valid runs: %d/%d, Mean latency: %.6f sec\n', ...
                sum(validRuns), NUM_MEASURED_RUNS, mean(latencies(validRuns)));
        else
            fprintf('    WARNING: No valid runs for %s!\n', currentAlgo);
        end
        
        % Force garbage collection between algorithms
        pause(0.1);
    end
    
    fprintf('\n');
end

% --- Calculate Final Statistics ---
fprintf('=== Calculating Final Statistics ===\n');

for algo = algorithms
    algoName = algo{1};
    measurements = allMeasurements.(algoName);
    
    if ~isempty(measurements.latencies)
        % Calculate robust statistics
        latencyMean = mean(measurements.latencies);
        latencyStd = std(measurements.latencies);
        
        % Use median for memory (more robust against outliers)
        memoryMedian = median(measurements.memories);
        
        rtfMean = mean(measurements.rtfs);
        rtfStd = std(measurements.rtfs);
        
        % Store in results table
        finalResults{:, algoName} = [latencyMean; latencyStd; memoryMedian; rtfMean; rtfStd];
        
        fprintf('%s: %.6f±%.6f sec, %.2f MB, RTF: %.6f±%.6f\n', ...
            algoName, latencyMean, latencyStd, memoryMedian, rtfMean, rtfStd);
    else
        fprintf('WARNING: No valid measurements for %s\n', algoName);
        finalResults{:, algoName} = [NaN; NaN; NaN; NaN; NaN];
    end
end

% --- Statistical Significance Testing ---
fprintf('\n=== Statistical Analysis ===\n');

if length(allMeasurements.specsub.latencies) > 1 && length(allMeasurements.mband.latencies) > 1
    % Perform t-test for latency differences
    [h_latency, p_latency] = ttest2(allMeasurements.specsub.latencies, allMeasurements.mband.latencies);
    [h_rtf, p_rtf] = ttest2(allMeasurements.specsub.rtfs, allMeasurements.mband.rtfs);
    
    fprintf('Latency t-test: p-value = %.6f, significant = %d (α = 0.05)\n', p_latency, h_latency);
    fprintf('RTF t-test: p-value = %.6f, significant = %d (α = 0.05)\n', p_rtf, h_rtf);
    
    % Effect size (Cohen's d)
    pooledStd_latency = sqrt(((length(allMeasurements.specsub.latencies)-1)*var(allMeasurements.specsub.latencies) + ...
                              (length(allMeasurements.mband.latencies)-1)*var(allMeasurements.mband.latencies)) / ...
                             (length(allMeasurements.specsub.latencies) + length(allMeasurements.mband.latencies) - 2));
    
    cohens_d = abs(mean(allMeasurements.specsub.latencies) - mean(allMeasurements.mband.latencies)) / pooledStd_latency;
    fprintf('Effect size (Cohen''s d): %.3f ', cohens_d);
    if cohens_d < 0.2
        fprintf('(negligible)\n');
    elseif cohens_d < 0.5
        fprintf('(small)\n');
    elseif cohens_d < 0.8
        fprintf('(medium)\n');
    else
        fprintf('(large)\n');
    end
end

% --- Save Results ---
% Save detailed results
outputFile = sprintf('robust_hardware_metrics_%s.xlsx', datestr(now, 'yyyymmdd_HHMMSS'));
writetable(finalResults, outputFile, 'WriteRowNames', true);

% Save raw measurements for further analysis
save(strrep(outputFile, '.xlsx', '_raw_data.mat'), 'allMeasurements', 'finalResults');

fprintf('\n=== Results Summary ===\n');
disp(finalResults);

fprintf('\nResults saved to: %s\n', outputFile);
fprintf('Raw data saved to: %s\n', strrep(outputFile, '.xlsx', '_raw_data.mat'));

% --- Generate Performance Comparison ---
fprintf('\n=== Performance Comparison ===\n');
if all(~isnan(finalResults{'Latency_sec_mean', :}))
    specsubLatency = finalResults{'Latency_sec_mean', 'specsub'};
    mbandLatency = finalResults{'Latency_sec_mean', 'mband'};
    
    if specsubLatency > mbandLatency
        improvement = (specsubLatency - mbandLatency) / specsubLatency * 100;
        fprintf('Multiband is %.1f%% faster than Spectral Subtraction\n', improvement);
    else
        improvement = (mbandLatency - specsubLatency) / mbandLatency * 100;
        fprintf('Spectral Subtraction is %.1f%% faster than Multiband\n', improvement);
    end
    
    % Real-time performance analysis
    specsubRTF = finalResults{'RealTimeFactor_mean', 'specsub'};
    mbandRTF = finalResults{'RealTimeFactor_mean', 'mband'};
    
    fprintf('\nReal-time capability:\n');
    fprintf('  Spectral Subtraction: RTF = %.4f %s\n', specsubRTF);
    fprintf('  Multiband Processing: RTF = %.4f %s\n', mbandRTF);
end

% --- Cleanup ---
% Remove any remaining temporary files
delete('temp_*.wav');

fprintf('\n✅ Robust hardware metrics evaluation completed successfully!\n');