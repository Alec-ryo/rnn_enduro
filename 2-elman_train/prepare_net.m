function net = prepare_net(num_back, num_hidden_layers, num_epochs, mod)

    net = elmannet(num_back, num_hidden_layers);
    % net.trainFcn='trainlm';
    net.divideFcn = '';

    net.inputs{1}.processFcns={};
    net.outputs{1,2}.processFcns={};
    
    if mod
        net.layers{1,1}.transferFcn='poslin';
        net.layers{2,1}.transferFcn='softmax';
    else
        for i=1:net.numLayers
            net.layers{i,1}.transferFcn='logsig';        
        end
    end
        
    net.trainParam.min_grad = 1e-20;
    net.trainParam.epochs = num_epochs;    
    net.trainParam.goal = 1e-5;
    net.trainParam.max_fail = 10;
    net.trainParam.lr=0.1;
    %net.trainParam.mu_max=1e99;
    net.efficiency.memoryReduction=4;
    net=init(net);
    
end