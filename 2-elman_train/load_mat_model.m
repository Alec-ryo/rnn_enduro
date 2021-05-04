model_path = 'models/play_45to49f_1to1000epoch10000h_500/playm_45to49f_1to1000epoch_10000h_500.mat';
load(model_path);

is_checkpoint = false;
use_train = false;

start_match = 45;
end_match = 45;

start_frame = 1;
end_frame = 1000;

obs = 'continue_training_';

%----------- Carregamento dos dados -----------%

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
    for k = start_frame:end_frame
        imageData = reshape(frames(k,:,:), [], 1);
        X_sample = [X_sample, imageData/255];
    end

    Y_sample = {};
    for idx = start_frame:end_frame
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

%----------- Cria nome do modelo -----------%
if mod
    model_name = strcat(obs, ...
                 'M', string(start_match), 'to', string(end_match), ...
                 'F', string(start_frame), 'to', string(end_frame), ...
                 'Epoch', string(num_epochs), ...
                 'H', string(num_hidden_layer), ...
                 'MOD');
else
    model_name = strcat(obs, ...
                 'M', string(start_match), 'to', string(end_match), ...
                 'F', string(start_frame), 'to', string(end_frame), ...
                 'Epoch', string(num_epochs), ...
                 'H', string(num_hidden_layer) );
end

disp(model_name);

%----------- Cria nome do modelo -----------%
if is_checkpoint
    net = checkpoint.net;
end

view(net);

%----------- Cria nome do modelo -----------%
if use_train
    net = train(net, X_train, Y_train, ...
                'useGPU', 'yes', ...
                'showResources','yes', ...
                'CheckpointFile', convertStringsToChars(model_name), ...
                'CheckpointDelay', 10);
end

Y = net(X_train);
plotconfusion(Y_train,Y);

% acc = show_accuracy(num_frames, Y, Y_train);
% show_predict_image(root_path, net, match, num_frames);