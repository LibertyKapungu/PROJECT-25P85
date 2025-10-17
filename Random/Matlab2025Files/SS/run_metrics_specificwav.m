% =========================================================================
% Run Metrics for Two Specific Algorithm Outputs
% =========================================================================

% --- Setup ---
addpath(genpath('objective_metrics/'));

% --- Reference clean speech file ---
ref_clean_speech = 'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\clean_reference.wav';

% --- List of processed speech files from different algorithms ---
processed_files = {
    'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\noisy_input.wav',
    'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\enhanced_GROUND_TRUTH_NOISE_ground_truth_mode_BANDS4_SPACINGLINEAR_FRAME8ms.wav',
    'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\enhanced_ESTIMATED_NOISE_standard_mode_BANDS4_SPACINGLINEAR_FRAME8ms.wav', 
    'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\mband_neural_vad_neural_guided_BANDS4_SPACINGLINEAR_FRAME8ms_NEURAL_W0.1.wav',
    'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\mband_neural_vad_neural_guided_BANDS4_SPACINGLINEAR_FRAME8ms_NEURAL_W0.3.wav',
    'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\mband_neural_vad_neural_guided_BANDS4_SPACINGLINEAR_FRAME8ms_NEURAL_W0.5.wav',
    'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\mband_neural_vad_neural_guided_BANDS4_SPACINGLINEAR_FRAME8ms_NEURAL_W0.7.wav',
    'C:\Users\gabi\Documents\University\Uni2025\Investigation\PROJECT-25P85\results\EXP2\spectral\NOISE_ESTIMATION\mband_neural_vad_neural_guided_BANDS4_SPACINGLINEAR_FRAME8ms_NEURAL_W0.9.wav',
};

% --- Corresponding algorithm names ---
algorithm_names = {
    'noisy_speech',
    'Ground_Truth',
    'Estimated_noise', 
    'mband_noisetry_01',
    'mband_noisetry_03',
    'mband_noisetry_05',
    'mband_noisetry_07',
    'mband_noisetry_09',
};

% --- Initialize results table ---
results = table();

% --- Loop through each processed file ---
for i = 1:length(processed_files)
    processed_speech = processed_files{i};
    algorithm = algorithm_names{i};
    
    fprintf('\n--- Metrics for %s ---\n', algorithm);
    
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

    % --- SII calculation (example values, adjust as needed) ---
    sp = [40 45 50 24 56 60 55 55 52 48 50 51 55 67 76 67 56 31];
    ns = [30 50 60 20 60 50 70 45 80 40 60 20 60 22 55 50 67 40];
    M = 5;
    SII_val = SII(sp, ns, M);

    % --- Store results ---
    result_row = table({algorithm}, {processed_speech}, snr_mean, segsnr_mean, wss_mean, llr_mean, ...
        is_mean, cep_mean, fwSNRseg, SIGv, BAKv, OVLv, SIGm, BAKm, OVLm, pesq_val, ...
        Csig, Cbak, Covl, NCM_val, CSh, CSm, CSl, SII_val, ...
        'VariableNames', {'Algorithm', 'File', 'SNR', 'SegSNR', 'WSS', 'LLR', ...
        'IS', 'CEP', 'fwSNRseg', 'SIGv', 'BAKv', 'OVLv', 'SIGm', 'BAKm', 'OVLm', ...
        'PESQ', 'Csig', 'Cbak', 'Covl', 'NCM', 'CSh', 'CSm', 'CSl', 'SII'});

    results = [results; result_row];
end

% --- Save results ---
writetable(results, 'selected_algorithm_metrics.xlsx');
fprintf('\n✅ Metrics saved to selected_algorithm_metrics.xlsx\n');