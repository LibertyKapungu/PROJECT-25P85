% =========================================================================
% Multi-Algorithm Batch Validation Script 
% =========================================================================

% --- Add relevant subfolders to MATLAB path ---
addpath(genpath('validation_dataset/noisy_speech/'))
addpath(genpath('spectral_algorithms/'))

% --- Define input folder and algorithms ---
inputFolder = 'validation_dataset/noisy_speech/';
algorithms = {'mband'};  % Add more algorithm names here

% --- Define function handles with optional parameters ---
% algorithmFuncs = containers.Map;
% algorithmFuncs('mband')   = @(infile, outfile) mband(infile, outfile, 4, 'log'); 

algorithmFuncs = containers.Map;
nbands = [4, 8];
freq_types = {'log', 'mel', 'linear'};

for b = 1:length(nbands)
    for f = 1:length(freq_types)
        key = sprintf('mband_B%d_%s', nbands(b), freq_types{f});
        algorithmFuncs(key) = @(infile, outfile) mband(infile, outfile, nbands(b), freq_types{f});
    end
end

algorithms = keys(algorithmFuncs);  % Automatically use all generated keys

% --- Check input folder exists ---
if ~exist(inputFolder, 'dir')
    error('Input directory "%s" does not exist.', inputFolder);
end

% --- Get all .wav files ---
files = dir(fullfile(inputFolder, '*.wav'));
if isempty(files)
    error('No .wav files found in input directory "%s".', inputFolder);
end

% --- Loop through each algorithm ---
for a = 1:length(algorithms)
    algorithm = algorithms{a};
    
    % --- Get function handle ---
    if ~isKey(algorithmFuncs, algorithm)
        error('No function defined for algorithm "%s".', algorithm);
    end
    func = algorithmFuncs(algorithm);
    
    % --- Create output folder if needed ---
    outdirectory = [algorithm '_processed_output/'];
    if ~exist(outdirectory, 'dir')
        mkdir(outdirectory);
        fprintf('Created output directory: %s\n', outdirectory);
    end

    fprintf('Processing with %s...\n', algorithm);
    for i = 1:numel(files)
        infile = fullfile(inputFolder, files(i).name);
        outfile = fullfile(outdirectory, [algorithm '_' files(i).name]);
        
        % --- Call the enhancement function dynamically ---
        func(infile, outfile);
        
        fprintf('Processed %d/%d: %s %s\n', i, numel(files), algorithm, files(i).name);
    end

    fprintf('%s processing complete\n\n', algorithm);
end

fprintf('=== All processing complete ===\n');
fprintf('Total files processed per algorithm: %d\n', numel(files));
