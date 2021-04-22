function [X_train, Y_train, num_frames] = prepare_data(root_path, match, num_frames)

    X_train = {};
    if num_frames == 'all'
        num_frames = numel(dir(strcat(root_path, 'img/match_', num2str(match), '/*.png')));
    end

    display(strcat('num_frames: ', int2str(num_frames)))

    for k = 1:num_frames
        jpgFilename = strcat(root_path, 'img/match_', num2str(match), '/', num2str(k-1), '.png');
        imageData = imread(jpgFilename);  
        imageData = reshape(imageData, [], 1);
        X_train = [X_train, imageData/255];
    end
    
    path = strcat(root_path, "txt/match_", num2str(match), "/actions.txt");
    fileID = fopen(path,'r');
    formatSpec = '%f';
    ftargets = fscanf(fileID, formatSpec);
    
    % targets = num2cell(reshape(ftargets, 1, 119));

    containing_actions = sort(unique(ftargets));
    containing_actions_size = size(containing_actions);

    Y_train = {};
    for idx = 1:num_frames
        one_hot_target = zeros(containing_actions_size(1), 1);
        pos = find(containing_actions == ftargets(idx));
        one_hot_target(pos) = 1;    
        Y_train = [Y_train, one_hot_target];
    end
    
end