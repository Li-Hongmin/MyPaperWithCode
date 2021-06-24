function [label, centers] = DnC_landmark(fea, p, selectN, unit)
    %DnC_landmark: 
    %   version 1.0 --April 2021
    %   Written by Hongmin Li (li.hongmin.xa@alumni.tsukuba.ac.jp)
    %===========

    [n, d] = size(fea);
    centers = zeros(p, d);
    sumD = ones(p, 1);
    obtainedSubsets = 1;
    label = ones(n, 1);
    warning off
    maxIter = 10;
    while obtainedSubsets < p
        labelNew = label;
        ks = full(sumD(1:obtainedSubsets));
        ks = set_sub_k(ks / sum(ks) * p);
        ks(ks > unit) = unit;

        for i = 1:obtainedSubsets

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
                [RestD, restLabel] = pdist2(curCenter, fea(restIdx, :),'euclidean', 'Smallest', 1);
                
                curLabel = zeros(1,curN);
                curLabel(randSample) = selectedLabel;
                curLabel(~randSample) = restLabel;
                % add the rest SumD
                if sum(ks) < p
                    restN = sum(restIdx);
                    curSumD = curSumD + RestD * sparse(1:restN, restLabel, 1, restN, k, restN);
                end
%                 clear selectedLabel restLabel randSample restN RestD
            elseif sum(indi) <= k
                k = min(sum(indi),2);
                [curLabel, curCenter, ~, curSumD] = litekmeans(fea(indi, :), k, 'MaxIter', maxIter);
            else
                [curLabel, curCenter, ~, curSumD] = litekmeans(fea(indi, :), k, 'MaxIter', maxIter);
            end

            % modify the local label and insert it into global one
            remainLabelIdx = curLabel == 1;
            curLabel(remainLabelIdx) = i;
            curLabel(~remainLabelIdx) = curLabel(~remainLabelIdx) + obtainedSubsets -1;
            labelNew(indi) = curLabel;
%             clear indi remainLabelIndx curLabel
            
            % insert centers and sumD into global one
            centers(i, :) = curCenter(1, :);
            centers(obtainedSubsets + 1:obtainedSubsets + k - 1, :) = curCenter(2:k, :);
%             clear curCenter
            sumD(i) = curSumD(1);
            sumD(obtainedSubsets + 1:obtainedSubsets + k - 1) = curSumD(2:k);
%             clear curSumD

            % count obtained clusters
            obtainedSubsets = obtainedSubsets + k -1;
        end

        label = labelNew;

    end

end
