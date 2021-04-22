function show_predict_image(net, num_frames, X_train)

    for k = 1:num_frames

        predicted = net(X_train(k));

        display(strcat('frame: ', num2str(k), ' -> action:', num2str(find(predicted{1} == max(predicted{1})))));

    end
    
end