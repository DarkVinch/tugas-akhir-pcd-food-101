imageFolder = 'food-101';
targetSize = [32 32];
inputSize = [32 32 3];
numClasses = 101;
numFiles = 0.1;
numTrainFiles = 0.90;
filterSize = 3;
numFilters = 3;
numChannels = 3;
maxEpochs = 10;

imds = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

[imdsUsed, imdsNotUsed] = splitEachLabel(imds, numFiles, 'randomized');
clear imdsNotUsed

[imdsTrain, imdsValidation] = splitEachLabel(imdsUsed, numTrainFiles, 'randomized');
clear imdsUsed

trainingData = augmentedImageDatastore(targetSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
validationData = augmentedImageDatastore(targetSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(filterSize, numFilters * 2, 'NumChannels', numChannels)   % 2-D CNN
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride = 2)
    convolution2dLayer(filterSize, numFilters * 4, 'NumChannels', numChannels * 2)   % 2-D CNN
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'MiniBatchSize', 1320, ...
    'Shuffle', 'never', ...
    'InitialLearnRate', 0.01);

net = trainNetwork(trainingData,layers,options);

YPred = classify(net, validationData);
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

labels = imdsValidation.Labels;

cm = confusionchart(YValidation, YPred);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
cm.Title = 'Food-101 Classification Confusion Matrix';

save cnnnet.mat net labels