%-------------- Configuracao ---------------%
mod = false;
obs = 'chuncked';

num_epochs = 5000;
num_hidden_layer = 200;

start_match = 45;
end_match = 50;

start_frame = 1;
end_frame = 1020;

use_gpu = false;

%----------- Cria nome do modelo -----------%
if mod
    model_name = strcat(obs, ...
                 '_m', string(start_match), 'to', string(end_match), ...
                 '_f', string(start_frame), 'to', string(end_frame), ...
                 '_epoch', string(num_epochs), ...
                 '_h', string(num_hidden_layer), ...
                 '_MOD');
else
    model_name = strcat(obs, ...
                 '_m', string(start_match), 'to', string(end_match), ...
                 '_f', string(start_frame), 'to', string(end_frame), ...
                 '_epoch', string(num_epochs), ...
                 '_h', string(num_hidden_layer) );
end

disp(model_name);

%----------- Carregamento dos dados -----------%
indices = { ...
           {0, 52, 62, 80, 106, 117, 137, 142, 156, 159, 166, 185, 215, 226, 235, 242, 252, 259, 279, 281, 292, 296, 299, ...
            303, 308, 315, 318, 321, 329, 333, 335, 354, 360, 374, 377, 382, 386, 390, 393, 402, 414, 420, 444, 447, ...
            465, 469, 487, 491, 504, 506, 510, 513, 525, 528, 538, 552, 558, 570, 576, 581, 584, 588, 595, 598, 604, ...
            607, 610, 618, 619, 622, 631, 641, 645, 662, 672, 685, 693, 697, 704, 718, 724, 726, 735, 739, 745, 754, ...
            755, 770, 773, 789, 790, 799, 805, 809, 823, 834, 850, 854, 857, 868, 874, 875, 877, 883, 885, 904, 906, ...
            911, 919, 923, 926, 932, 943, 962, 965, 984, 991, 999, 1007, 1010, 1015, 1017, 1018}, ...
           {0, 31, 45, 47, 56, 63, 69, 85, 88, 92, 96, 112, 118, 124, 130, 137, 143, 151, 158, 170, 184, 202, 223, 243, 249, ...
            255,260,269,272,275,276,282,285,295,296,297,306,308,318,322,331,334,337,338,341,343,354,357,381,387,400,402,408, ...
            420,423,426,429,441,443,445,448,454,463,465,474,481,502,506,524,533,538,548,556,562,566,570,581,588,603,604,609, ...
            652,668,681,698,710,714,725,729,736,739,743,748,749,752,766,770,774,778,781,792,795,803,804,806,814,818,821,828, ...
            834,837,843,849,855,862,868,874,884,888,895,896,919,935,942,945,957,958,963,964,965,977,980,984,989,997,1000}, ...
           {0,37,53,81,94,99,119,123,124,142,144,146,162,193,211,213,222,230,232,235,238,240,252,255,258,267,270,275, ...
            278,284,287,289,300,307,313,331,352,359,371,379,386,389,392,409,412,432,435,440,442,446,462,467,472, ...
            506,527,531,535,553,557,561,571,579,589,602,607,609,612,616,621,631,634,638,640,644,645,661,669,673,676, ...
            692,695,698,701,711,717,720,721,734,744,747,750,755,759,774,785,793,807,859,891,895,912,927,948,958,969, ...
            980,997,998,1004}, ...
           {0,33,58,75,86,100,118,120,139,142,147,159,162,165,167,174,179,180,183,192,197,199,209,214,217,222,227,231, ...
            232,239,244,251,262,266,271,275,277,282,287,291,294,297,308,311,316,318,322,327,339,342,348,352,362,372, ...
            383,397,399,402,405,417,420, 425,428,454,457,464,470,477,481,485,488,494,495,503,507,517,535,537,540,553, ...
            556,558,577,588,598,601,609,620,634,638,647,650,656,659,660,663,669,672,681,684,696,702,705,708,716,719,726, ...
            730,739,743,746,758,765,766,777,783,785,786,790,792,800,802,808,809,810,816,819,823,827,835,837,842,845,853, ...
            865,868,873,885,889,891,897,904,906,918,941,949,956,961,969,973,976,981,984,991,996,999,1002}, ...
           {0,59,68,73,76,86,94,97,101,104,109,118,120,123,128,134,136,139,143,145,148,154,160,162,177,190,195,199,204, ...
            208,212,220,221,222,232,234,246,251,253,256,263,270,273,287,302,314,322,355,357,368,396,397,401,416,428,433, ...
            437,442,446,452,456,469,475,490,500,506,525,526,538,545,551,554,558,562,571,579,582,586,587,600,612,664,823, ...
            824,832,850,858,870,879,881,889,898,901,913,916,924,932,936,943,952,955,975,985,992,997,1001}, ...
           {0,36,56,94,114,121,136,140,143,153,159,162,171,174,182,190,197,201,215,219,237,239,249,251,254,265,269,297,322, ...
            335,340,343,348,352,355,363,368,371,375,384,388,395,405,411,439,461,464,468,477,483,485,487,494,497,501,503, ...
            514,519,521,526,529,532,538,547,551,555,560,579,584,597,600,602,607,609,617,623,626,634,638,644,649,652,663, ...
            671,672,675,678,685,688,699,700,712,719,724,726,735,744,753,756,775,797,806,812,818,826,829,832,848,852,856, ...
            866,869,872,875,881,886,891,893,895,901,910,930,936,940,953,958,989,997,1004,1006}  ...
          };

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
    
    m_indices = indices{m - 44};
    tam_indices = size(m_indices);
    
    for chunk = 1:tam_indices(2) - 1 % 123
        
        start_chunk = m_indices{chunk} + 1;
        
        end_chunk = m_indices{chunk + 1};
        
        disp('--------------')
        disp(m);
        disp(start_chunk);
        disp(end_chunk);
        
        X_sample = {};
        for k = start_chunk:end_chunk
            imageData = reshape(frames(k,:,:), 170, 120);
            imageData = imageData(30:129, :);
            imageData = reshape(imageData, [], 1);
            X_sample = [X_sample, imageData/255];
        end

        Y_sample = {};
        for idx = start_chunk:end_chunk
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
end

del X_sample;
del Y_sample;

disp('preparing network');
net = prepare_net(1, num_hidden_layer, num_epochs, mod);

disp('training');

net.trainParam.showWindow = 0;
if use_gpu
    [net, tr] = train(net, X_train, Y_train, ...
                'useGPU', 'yes', ...
                'showResources','no', ...
                'CheckpointFile', convertStringsToChars(model_name), ...
                'CheckpointDelay', 10);
else
    [net, tr] = train(net, X_train, Y_train, ...
                'useGPU', 'no', ...
                'showResources','no', ...
                'CheckpointFile', convertStringsToChars(model_name), ...
                'CheckpointDelay', 10);
end

% disp('predicting');
% Y = net(X_train);
% 
% plotconfusion(Y_train,Y);

%show_predict_image(data_path, net, match, num_frames);

save(model_name);