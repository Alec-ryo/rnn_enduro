% data_path = '/home/ryo/Desktop/tcc/workspace/1-generate/data/match_38/mat/data.mat';
% 
% disp('loading data');
% load(data_path)
% 
% X_train = {};
% for k = 1:num_frames
%     imageData = reshape(frames(k,:,:), [], 1);
%     X_train = [X_train, imageData/255];
% end
% 
% containing_actions = sort(unique(actions));
% containing_actions_size = size(containing_actions);
% 
% Y_train = {};
% for idx = 1:num_frames
%     one_hot_target = zeros(length(containing_actions), 1);
%     pos = find(containing_actions == actions(idx));
%     one_hot_target(pos) = 1;    
%     Y_train = [Y_train, one_hot_target];
% end

all_data = {};
all_targets = {};
num_hidden_layer = 200;

data_path = strcat('../1-generate/data/match_45/mat/data.mat');
load(data_path);
containing_actions = sort(unique(actions));
containing_actions_size = size(containing_actions);

X_train = {};
Y_train = {};

for n_seq = 45:50
    
    % X_train
    data_path = strcat('../1-generate/data/match_', string(n_seq),'/mat/data.mat');
    load(data_path);
    
    frames = frames(1:1000,:);
    size_of_frames = size(frames);
    reshaped_frames = reshape(frames, size_of_frames(2), size_of_frames(1));
    
    X_train = [X_train, reshaped_frames/255];
    
    % Y_train    
    one_hot_targets = zeros(length(containing_actions), 1000);

    for images = 1:1000

        pos = find(containing_actions == actions(images));
        one_hot_targets(pos, images) = 1;    

    end
    
    Y_train = [Y_train, one_hot_targets];
    
end

disp('preparing network');
net = prepare_net(1, num_hidden_layer);

disp('training');
net = train(net, X_train, Y_train);
view(net);

disp('predicting');
Y = net(X_train);

plotconfusion(Y_train,Y);

%show_predict_image(data_path, net, match, num_frames);

save('net_model');