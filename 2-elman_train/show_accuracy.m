function acc = show_accuracy(num_frames, predicted, targets)

    acerto = 0;
    for i = 1:num_frames
        if find(predicted{i} == max(predicted{i})) == find(targets{i} == max(targets{i}))
            acerto = acerto + 1;
        end
    end
    acc = (acerto / 119)*100;

end