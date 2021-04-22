% root_path = '..\model_matlab\zigzag_model.mat';

% net = load('../model_matlab/zigzag/zigzag_model.mat');
% exportONNXNetwork(net,'onnx_model.onnx');

% modelfile = 'model.h5';
% net = importKerasNetwork(modelfile)

% model_path = 'C:\Users\alece\Desktop\UnB\tcc\matlab\model_matlab\zigzag\net_model.mat';
model_path = 'net_model.mat';
load(model_path);
Y = net(X_train);
plotconfusion(Y_train,Y);
% acc = show_accuracy(num_frames, Y, Y_train);

show_predict_image(root_path, net, match, num_frames);