function net = prepare_net(num_back, num_hidden_layers)

    net = elmannet(num_back, num_hidden_layers);
    % net.trainFcn='traingdm';
    net.divideFcn = '';

    net.inputs{1}.processFcns={};
    net.outputs{1,2}.processFcns={};
    
    for i=1:net.numLayers
        net.layers{i,1}.transferFcn='logsig';        
    end

    net.trainParam.min_grad = 1e-20;
    net.trainParam.epochs = 3000;    
    net.trainParam.goal = 1e-5;
    net.trainParam.max_fail = 10;
    net.trainParam.lr=0.1;
    %net.trainParam.mu_max=1e99;
    net.efficiency.memoryReduction=4;
    net=init(net);
    
end