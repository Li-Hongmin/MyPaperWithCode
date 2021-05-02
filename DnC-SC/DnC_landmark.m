function [label, centers] = DnC_landmark(fea, p, selectN, unit)
    [n, d] = size(fea);
    if d <100
        distance = @pdist2;
    else
        distance = @(x,y) EuDist2(x,y,0);
    end
    centers = zeros(p, d);
    sumD = ones(p, 1);
    obtainedClusters = 1;
    label = ones(n, 1);
    warning off
    maxIter = 10;
    while obtainedClusters < p
        labelNew = label;
        ks = full(sumD(1:obtainedClusters));
        ks = set_sub_k(ks / sum(ks) * p);
        ks(ks > unit) = unit;

        for i = 1:obtainedClusters

            k = ks(i);

            if k == 1
                continue
            end

            indi = label == i;
            curN = sum(indi);

            if curN > selectN

                % light-k-means

                % random sampling
                randSample = false(1, curN);
                randSample(randperm(curN, selectN)) = true;
                [selectedIdx, restIdx] = deal(indi);
                selectedIdx(indi) = randSample;
                restIdx(indi) = ~randSample;
                
                % obtain centers
                [selectedLabel, curCenter, ~, curSumD] = litekmeans(fea(selectedIdx, :), k, 'MaxIter', maxIter);

                % insert the rest samples
                D = distance(fea(restIdx, :), curCenter);
                [RestD, restLabel] = min(D, [], 2); clear D
                curLabel = zeros(1,curN);
                curLabel(randSample) = selectedLabel;
                curLabel(~randSample) = restLabel;
                % add the rest SumD
                if sum(ks) < p
                    restN = sum(restIdx);
                    curSumD = curSumD + RestD' * sparse(1:restN, restLabel, 1, restN, k, restN);
                end
                clear selectedLabel restLabel randSample restN RestD
            elseif sum(indi) <= k
                k = min(sum(indi),2);
                [curLabel, curCenter, ~, curSumD] = litekmeans(fea(indi, :), k, 'MaxIter', maxIter);
            else
                [curLabel, curCenter, ~, curSumD] = litekmeans(fea(indi, :), k, 'MaxIter', maxIter);
            end

            % modify the local label and insert it into global one
            remainLabelIdx = curLabel == 1;
            curLabel(remainLabelIdx) = i;
            curLabel(~remainLabelIdx) = curLabel(~remainLabelIdx) + obtainedClusters -1;
            labelNew(indi) = curLabel;
            clear indi remainLabelIndx curLabel
            
            % insert centers and sumD into global one
            centers(i, :) = curCenter(1, :);
            centers(obtainedClusters + 1:obtainedClusters + k - 1, :) = curCenter(2:k, :);
            clear curCenter
            sumD(i) = curSumD(1);
            sumD(obtainedClusters + 1:obtainedClusters + k - 1) = curSumD(2:k);
            clear curSumD

            % count obtained clusters
            obtainedClusters = obtainedClusters + k -1;
        end

        label = labelNew;

    end

end
