model_path = 'play_m45to50_f1to1000_epoch10000_h200.mat';
load(model_path);

%-------------- Configuracao ---------------%

start_match = 45;
end_match = 50;

start_frame = 1;
end_frame = 1000;

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

%----------- train -----------%

disp('train more');
[net, tr] = train(checkpoint.net, X_train, Y_train, ...
      'useGPU', 'no', ...
      'showResources','no', ...
      'CheckpointFile', 'enduro_train_more_3811_h200', ...
      'CheckpointDelay', 10);

dst_path = 'adapted_model/new_model_2'
save(dst_path);

plotconfusion(Y_train,Y);