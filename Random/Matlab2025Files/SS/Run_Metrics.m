% =========================================================================
% Run Metrics for Multiple Algorithms
% =========================================================================

% --- Setup ---
addpath(genpath('objective_metrics/'));
addpath(genpath('validation_dataset/'));

% --- Reference clean speech file ---
ref_clean_speech = 'validation_dataset/clean_speech/S_56_02.wav';

% --- List of algorithm output folders ---
algorithm_dirs = {'mband_processed_output', 'python_mband_processed_output'};  % Add more if needed

% --- Initialize results table ---
results = table();

% --- Loop through each algorithm folder ---
for a = 1:length(algorithm_dirs)
    algorithm = algorithm_dirs{a};
    
    if ~exist(algorithm, 'dir')
        warning('Directory "%s" does not exist. Skipping.', algorithm);
        continue;
    end
    
    files = dir(fullfile(algorithm, '*.wav'));
    if isempty(files)
        warning('No .wav files found in "%s". Skipping.', algorithm);
        continue;
    end
    
    fprintf('=== Evaluating files in %s ===\n', algorithm);
    
    for i = 1:numel(files)
        processed_speech = fullfile(algorithm, files(i).name);
        fprintf('\n--- Metrics for %s (%d/%d) ---\n', files(i).name, i, numel(files));
        
        % --- Compute metrics ---
        [snr_mean, segsnr_mean] = comp_snr(ref_clean_speech, processed_speech);
        wss_mean  = comp_wss(ref_clean_speech, processed_speech);
        llr_mean  = comp_llr(ref_clean_speech, processed_speech);
        is_mean   = comp_is(ref_clean_speech, processed_speech);
        cep_mean  = comp_cep(ref_clean_speech, processed_speech);
        fwSNRseg  = comp_fwseg(ref_clean_speech, processed_speech);
        [SIGv, BAKv, OVLv] = comp_fwseg_variant(ref_clean_speech, processed_speech);
        [SIGm, BAKm, OVLm] = comp_fwseg_mars(ref_clean_speech, processed_speech);
        pesq_val = pesq(ref_clean_speech, processed_speech);
        [Csig, Cbak, Covl] = composite(ref_clean_speech, processed_speech);
        NCM_val = NCM(ref_clean_speech, processed_speech);
        [CSh, CSm, CSl] = CSII(ref_clean_speech, processed_speech);
        
        sp = [40 45 50 24 56 60 55 55 52 48 50 51 55 67 76 67 56 31];
        ns = [30 50 60 20 60 50 70 45 80 40 60 20 60 22 55 50 67 40];
        M = 5;
        SII_val = SII(sp, ns, M);
        
        % --- Store results in a row ---
        result_row = table({algorithm}, {files(i).name}, snr_mean, segsnr_mean, wss_mean, llr_mean, ...
            is_mean, cep_mean, fwSNRseg, SIGv, BAKv, OVLv, SIGm, BAKm, OVLm, pesq_val, ...
            Csig, Cbak, Covl, NCM_val, CSh, CSm, CSl, SII_val, ...
            'VariableNames', {'Algorithm', 'File', 'SNR', 'SegSNR', 'WSS', 'LLR', ...
            'IS', 'CEP', 'fwSNRseg', 'SIGv', 'BAKv', 'OVLv', 'SIGm', 'BAKm', 'OVLm', ...
            'PESQ', 'Csig', 'Cbak', 'Covl', 'NCM', 'CSh', 'CSm', 'CSl', 'SII'});
        
        % --- Append to results table ---
        results = [results; result_row];
    end
end

% --- Save results to Excel ---
writetable(results, 'all_algorithm_metrics.xlsx');
fprintf('\n✅ All metrics saved to all_algorithm_metrics.xlsx\n');
