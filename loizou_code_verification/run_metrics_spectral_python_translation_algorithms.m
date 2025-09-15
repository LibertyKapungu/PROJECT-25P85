% =========================================================================
% Run Metrics for Spectral Type Algorithm
% =========================================================================
% This script evaluates Spectral-processed speech against clean reference S_56_02.wav
% -------------------------------------------------------------------------

% --- Add metric subfolders to MATLAB path ---
addpath(genpath('objective_metrics/'));
addpath(genpath('validation_dataset/'));
addpath(genpath('spectral_processed_output/'))

% --- Reference files ---
ref_clean_speech_spectral = 'validation_dataset/clean_speech/S_56_02.wav'; % Clean reference for Spectral-processed files

% =========================================================================
% Evaluate Spectral-processed files against S_56_02.wav
% =========================================================================
fprintf('=== SCENARIO 1: Evaluating Spectral-processed files ===\n');

spectral_processed_directory = 'spectral_processed_output/';
if ~exist(spectral_processed_directory, 'dir')
    error('Processed directory "%s" does not exist.', spectral_processed_directory);
end

files = dir(fullfile(spectral_processed_directory, '*.wav'));
if isempty(files)
    error('No .wav files found in "%s".', spectral_processed_directory);
end

% --- Loop through Spectral-processed files ---
for i = 1:numel(files)
    processed_speech = fullfile(spectral_processed_directory, files(i).name);
    
    % Use S_56_02.wav as clean reference
    clean_speech = ref_clean_speech_spectral;
    
    fprintf('\n ========== Obtaining Metrics for %d/%d: %s ========== \n', i, numel(files), files(i).name);

    % ---------- Quality ----------- %
    [snr_mean, segsnr_mean] = comp_snr(clean_speech, processed_speech)
    wss_mean  = comp_wss(clean_speech, processed_speech)
    llr_mean  = comp_llr(clean_speech, processed_speech)
    is_mean   = comp_is(clean_speech, processed_speech)
    cep_mean  = comp_cep(clean_speech, processed_speech)
    fwSNRseg  = comp_fwseg(clean_speech, processed_speech)
    [SIGv, BAKv, OVLv] = comp_fwseg_variant(clean_speech, processed_speech)
    [SIGm, BAKm, OVLm] = comp_fwseg_mars(clean_speech, processed_speech)
    pesq_val = pesq(clean_speech, processed_speech)
    [Csig, Cbak, Covl] = composite(clean_speech, processed_speech)

    % ---------- Intelligibility ----------- %
    NCM_val = NCM(clean_speech, processed_speech)
    
    [CSh, CSm, CSl] = CSII(clean_speech, processed_speech)
    
    sp=[40 45 50 24 56 60 55 55 52 48 50 51 55 67 76 67 56 31]; % values in dB SPL
    ns=[30 50 60 20 60 50 70 45 80 40 60 20 60 22 55 50 67 40]; % values in dB SPL
    M= 5;
    SII_val = SII(sp, ns, M)

    fprintf('Processed Spectral %d/%d: %s\n', i, numel(files), files(i).name);

end
