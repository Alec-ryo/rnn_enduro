model_path = 'models/playM45to45F1to1000Epoch5000H200/playM45to45F1to1000Epoch5000H200.mat';
load(model_path);

net = train(net, X_train, Y_train);

dst_path = 'adapted_model/new_model'
save(dst_path);