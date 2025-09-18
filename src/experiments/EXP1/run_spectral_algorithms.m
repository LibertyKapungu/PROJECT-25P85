% =========================================================================
% Wiener Batch Validation Script
% =========================================================================
% This script processes all .wav files from a given input folder using the
% Wiener type algorithms described in Speech Enhancement: Theory and 
% Practice, Second Edition by Philipos C. Loizou, and saves the processed 
% results into a specified output folder. 
% -------------------------------------------------------------------------

% --- Add relevant subfolders to MATLAB path ---
addpath(genpath('validation_dataset/noisy_speech/'))
addpath(genpath('spectral_algorithms/'))
addpath('spectral_processed_output/')

inputFolder = 'validation_dataset/noisy_speech/';   % Folder with input .wav files
outdirectory = 'spectral_processed_output/';  % Output folder for processed files

% --- Check input directory exists ---
if ~exist(inputFolder, 'dir')
    error('Input directory "%s" does not exist.', inputFolder);
end

% --- Check output directory exists ---
if ~exist(outdirectory, 'dir')
    error('Output directory "%s" does not exist.', outdirectory);
end

% --- Get all .wav files in the folder ---
files = dir(fullfile(inputFolder, '*.wav'));

% --- Error if no .wav files found ---
if isempty(files)
    error('No .wav files found in input directory "%s".', inputFolder);
end

% --- Process each file using specsub ---
fprintf('Processing noisy speech with specsub...\n');
for i = 1:numel(files)
    infile = fullfile(inputFolder, files(i).name);
    outfile = fullfile(outdirectory, ['specsub_' files(i).name]);
    specsub(infile, outfile);
    fprintf('Processed %d/%d: specsub %s\n', i, numel(files), files(i).name);
end
fprintf('specsub processing complete\n\n');

fprintf('=== All Spectral processing complete ===\n');
fprintf('Total files processed: %d\n', numel(files));
fprintf('Output directory: %s\n', outdirectory);