num_hidden_layer = 200;
start_match = 45;
end_match = 46;

disp('loading data');

data_path = strcat('../1-generate/data/match_', string(start_match), '/mat/data.mat');
load(data_path);

containing_actions = sort(unique(actions));
containing_actions_size = size(containing_actions);

train_exist = false;

for m = start_match:end_match

    data_path = strcat('../1-generate/data/match_', string(m), '/mat/data.mat');

    load(data_path);

    X_sample = {};
    for k = 1:1000
        imageData = reshape(frames(k,:,:), [], 1);
        X_sample = [X_sample, imageData/255];
    end

    Y_sample = {};
    for idx = 1:1000
        one_hot_target = zeros(length(containing_actions), 1);
        pos = find(containing_actions == actions(idx));
        one_hot_target(pos) = 1;    
        Y_sample = [Y_sample, one_hot_target];
    end

    if ( train_exist )
        X_train = catsamples(X_train, X_sample, 'pad');
        Y_train = catsamples(Y_train, Y_sample, 'pad'); 
    else
        X_train = X_sample;
        Y_train = Y_sample; 
        train_exist = true;
    end
end

disp('preparing network');
useSig = true;
net = prepare_net(1, num_hidden_layer, useSig);

disp('training');
net = train(net, X_train, Y_train);
% net = train(net, X_train, Y_train, ...
%       'CheckpointFile', 'enduro_elman_epoch5000H200SigSig', ...
%       'CheckpointDelay', 10);
  
% view(net);
% 
% disp('predicting');
% Y = net(X_train);
% 
% plotconfusion(Y_train,Y);

%show_predict_image(data_path, net, match, num_frames);

save( strcat('zigzag_epoch5000H', string(num_hidden_layer), 'SigSig') );