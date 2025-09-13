% =========================================================================
% Run Metrics for Wiener Type Algorithm - FIXED WITH SII
% =========================================================================
% This script evaluates reference noisy speech against clean reference 
% S_03_01.wav.
% -------------------------------------------------------------------------

% --- Add metric subfolders to MATLAB path ---
addpath(genpath('objective_metrics/'));
addpath(genpath('validation_dataset/'));

% --- Reference files ---
ref_noise_speech = 'validation_dataset/reference/S_03_01_babble_sn0_klt.wav'; % Noisy reference file
ref_clean_speech = 'validation_dataset/reference/S_03_01.wav'; % Clean reference for noisy reference file

% =========================================================================
% Evaluate noisy reference against S_03_01.wav
% =========================================================================
fprintf('\n === Evaluating noisy reference file === \n');

clean_speech = ref_clean_speech;
processed_speech = ref_noise_speech;

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

sp=[40 45 50 24 56 60 55 55 52 48 50 51 55 67 76 67 56 31];
ns=[30 50 60 20 60 50 70 45 80 40 60 20 60 22 55 50 67 40];
M= 5;
SII_val = SII(sp, ns, M)

fprintf('Processed noisy reference: %s\n', ref_noise_speech);