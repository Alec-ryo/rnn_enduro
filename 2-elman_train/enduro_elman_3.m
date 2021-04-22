%-------------- Configuracao ---------------%
mod = false;
obs = 'play';

num_epochs = 5000;
num_hidden_layer = 500;

start_match = 45;
end_match = 45;

start_frame = 1;
end_frame = 1000;

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

disp('preparing network');
net = prepare_net(1, num_hidden_layer, num_epochs, mod);

disp('training');
net = train(net, X_train, Y_train);
% net = train(net, X_train, Y_train, ...
%       'CheckpointFile', 'enduro_elman_epoch5000H200SigSig', ...
%       'CheckpointDelay', 10);

% disp('predicting');
% Y = net(X_train);
% 
% plotconfusion(Y_train,Y);

%show_predict_image(data_path, net, match, num_frames);

save(model_name);