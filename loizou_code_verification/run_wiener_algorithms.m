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
addpath(genpath('wiener_algorithms/'))
addpath('wiener_processed_output/')

inputFolder = 'validation_dataset/noisy_speech/';   % Folder with input .wav files
outdirectory = 'wiener_processed_output/';  % Output folder for processed files

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

% --- Process each file using wiener_as ---
fprintf('Processing noisy speech with wiener_as...\n');
for i = 1:numel(files)
    infile = fullfile(inputFolder, files(i).name);
    outfile = fullfile(outdirectory, ['wiener_as_' files(i).name]);
    wiener_as(infile, outfile);
    fprintf('Processed %d/%d: wiener_as_%s\n', i, numel(files), files(i).name);
end
fprintf('wiener_as processing complete\n\n');

% % --- Process each file using wiener_wt ---
% fprintf('Processing noisy speech with wiener_wt...\n');
% for i = 1:numel(files)
%     infile = fullfile(inputFolder, files(i).name);
%     outfile = fullfile(outdirectory, ['wiener_wt_' files(i).name]);
%     wiener_wt(infile, outfile);
%     fprintf('Processed %d/%d: wiener_wt_%s\n', i, numel(files), files(i).name);
% end
% fprintf('wiener_wt processing complete\n\n');
% 
% % --- Process each file using wiener_iter ---
% number_of_iterations = 3;
% fprintf('Processing noisy speech with wiener_iter (%d iterations)...\n', number_of_iterations);
% for i = 1:numel(files)
%     infile = fullfile(inputFolder, files(i).name);
%     outfile = fullfile(outdirectory, ['wiener_iter_' files(i).name]);
%     wiener_iter(infile, outfile, number_of_iterations);
%     fprintf('Processed %d/%d: wiener_iter_%s\n', i, numel(files), files(i).name);
% end
% fprintf('wiener_iter processing complete\n\n');
% 
% % --- Process each file using audnoise ---
% fprintf('Processing noisy speech with audnoise...\n');
% for i = 1:numel(files)
%     infile = fullfile(inputFolder, files(i).name);
%     outfile = fullfile(outdirectory, ['audnoise_' files(i).name]);
%     audnoise(infile, outfile);  % Custom Wiener filtering function
%     fprintf('Processed %d/%d: audnoise_%s\n', i, numel(files), files(i).name);
% end
% fprintf('audnoise processing complete\n\n');

fprintf('=== All Wiener processing complete ===\n');
fprintf('Total files processed: %d\n', numel(files));
fprintf('Output directory: %s\n', outdirectory);