% =========================================================================
% Run Metrics for Wiener Type Algorithm - FIXED WITH SII
% =========================================================================
% This script evaluates two scenarios:
% 1. Wiener-processed speech against clean reference S_56_02.wav
% 2. Reference noisy speech against clean reference S_03_01.wav
% Results are stored in separate tables and exported to separate CSV files.
% -------------------------------------------------------------------------

% --- Add metric subfolders to MATLAB path ---
addpath(genpath('objective_metrics/'));
addpath(genpath('validation_dataset/'));
addpath(genpath('wiener_processed_output/'))

% --- Reference files ---
ref_clean_speech_wiener = 'validation_dataset/clean_speech/S_56_02.wav'; % Clean reference for Wiener-processed files
ref_noise_speech = 'validation_dataset/reference/S_03_01_babble_sn0_klt.wav'; % Noisy reference file
ref_clean_speech_noisy = 'validation_dataset/reference/S_03_01.wav'; % Clean reference for noisy reference file

% --- Column names ---
column_names = {...
    'Method', 'Filename', ...
    'SNR_mean', 'SegSNR_mean', 'WSS_mean', 'LLR_mean', 'IS_mean', 'CEP_mean', 'fwSNRseg', ...
    'SIGv', 'BAKv', 'OVLv', 'SIGm', 'BAKm', 'OVLm', ...
    'PESQ', 'Csig', 'Cbak', 'Covl', ...
    'NCM', 'CSII_high', 'CSII_mid', 'CSII_low', 'SII'};

% =========================================================================
% SCENARIO 1: Evaluate Wiener-processed files against S_56_02.wav
% =========================================================================
fprintf('=== SCENARIO 1: Evaluating Wiener-processed files ===\n');

wiener_processed_directory = 'wiener_processed_output/';
if ~exist(wiener_processed_directory, 'dir')
    error('Processed directory "%s" does not exist.', wiener_processed_directory);
end

files = dir(fullfile(wiener_processed_directory, '*.wav'));
if isempty(files)
    error('No .wav files found in "%s".', wiener_processed_directory);
end

% --- Initialize results cell array for Wiener files ---
wiener_results = cell(numel(files), 24);

% --- Loop through Wiener-processed files ---
for i = 1:numel(files)
    processed_speech = fullfile(wiener_processed_directory, files(i).name);
    
    try
        % Use S_56_02.wav as clean reference
        clean_speech = ref_clean_speech_wiener;
        
        % ---------------- Quality Metrics ----------------
        [snr_mean, segsnr_mean] = comp_snr(clean_speech, processed_speech);
        wss_mean  = comp_wss(clean_speech, processed_speech);
        llr_mean  = comp_llr(clean_speech, processed_speech);
        is_mean   = comp_is(clean_speech, processed_speech);
        cep_mean  = comp_cep(clean_speech, processed_speech);
        fwSNRseg  = comp_fwseg(clean_speech, processed_speech);
        
        % Frequency-weighted segmental measures
        [SIGv, BAKv, OVLv] = comp_fwseg_variant(clean_speech, processed_speech);
        [SIGm, BAKm, OVLm] = comp_fwseg_mars(clean_speech, processed_speech);
        
        % PESQ measure
        pesq_val = pesq(clean_speech, processed_speech);
        
        % Composite quality scores
        [Csig, Cbak, Covl] = composite(clean_speech, processed_speech);
        
        % ---------------- Intelligibility Metrics ----------------
        NCM_val = NCM(clean_speech, processed_speech);
        [CSh, CSm, CSl] = CSII(clean_speech, processed_speech);
        
        % NEW: SII measure (requires level vectors)
        [sp, ns, M] = extract_levels_for_SII(clean_speech, processed_speech);
        SII_val = SII(sp, ns, M);
        
        % ---------------- Store results for Wiener files ----------------
        wiener_results = store_results(wiener_results, i, 'Wiener', files(i).name, ...
            snr_mean, segsnr_mean, wss_mean, llr_mean, is_mean, cep_mean, fwSNRseg, ...
            SIGv, BAKv, OVLv, SIGm, BAKm, OVLm, ...
            pesq_val, Csig, Cbak, Covl, NCM_val, CSh, CSm, CSl, SII_val);
        
        fprintf('Processed Wiener %d/%d: %s\n', i, numel(files), files(i).name);
        
    catch Err
        fprintf('Error processing Wiener file %s: %s\n', files(i).name, Err.message);
        wiener_results = store_failed_results(wiener_results, i, 'Wiener', files(i).name);
    end
end

% =========================================================================
% SCENARIO 2: Evaluate noisy reference against S_03_01.wav
% =========================================================================
fprintf('\n=== SCENARIO 2: Evaluating noisy reference file ===\n');

% --- Initialize results cell array for noisy reference (single file) ---
noisy_results = cell(1, 24);

try
    clean_speech = ref_clean_speech_noisy;
    processed_speech = ref_noise_speech;
    
    [snr_mean, segsnr_mean] = comp_snr(clean_speech, processed_speech);
    wss_mean  = comp_wss(clean_speech, processed_speech);
    llr_mean  = comp_llr(clean_speech, processed_speech);
    is_mean   = comp_is(clean_speech, processed_speech);
    cep_mean  = comp_cep(clean_speech, processed_speech);
    fwSNRseg  = comp_fwseg(clean_speech, processed_speech);
    
    [SIGv, BAKv, OVLv] = comp_fwseg_variant(clean_speech, processed_speech);
    [SIGm, BAKm, OVLm] = comp_fwseg_mars(clean_speech, processed_speech);
    
    pesq_val = pesq(clean_speech, processed_speech);
    [Csig, Cbak, Covl] = composite(clean_speech, processed_speech);
    
    NCM_val = NCM(clean_speech, processed_speech);
    [CSh, CSm, CSl] = CSII(clean_speech, processed_speech);
    
    % NEW: SII measure
    [sp, ns, M] = extract_levels_for_SII(clean_speech, processed_speech);
    SII_val = SII(sp, ns, M);
    
    noisy_results = store_results(noisy_results, 1, 'Noisy_Reference', ref_noise_speech, ...
        snr_mean, segsnr_mean, wss_mean, llr_mean, is_mean, cep_mean, fwSNRseg, ...
        SIGv, BAKv, OVLv, SIGm, BAKm, OVLm, ...
        pesq_val, Csig, Cbak, Covl, NCM_val, CSh, CSm, CSl, SII_val);
    
    fprintf('Processed noisy reference: %s\n', ref_noise_speech);
    
catch Err
    fprintf('Error processing noisy reference %s: %s\n', ref_noise_speech, Err.message);
end

% =========================================================================
% Export results to separate CSV files
% =========================================================================

results_dir = 'results';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
    fprintf('Created results directory: %s\n', results_dir);
end

wiener_filename = fullfile(results_dir, 'wiener_evaluation_results.csv');
export_to_csv(wiener_results, column_names, wiener_filename);
fprintf('Wiener evaluation results saved to %s\n', wiener_filename);

noisy_filename = fullfile(results_dir, 'noisy_reference_evaluation_results.csv');
export_to_csv(noisy_results, column_names, noisy_filename);
fprintf('Noisy reference evaluation results saved to %s\n', noisy_filename);

fprintf('\n=== Evaluation Complete ===\n');
fprintf('Wiener files evaluated against: %s\n', ref_clean_speech_wiener);
fprintf('Noisy reference evaluated against: %s\n', ref_clean_speech_noisy);