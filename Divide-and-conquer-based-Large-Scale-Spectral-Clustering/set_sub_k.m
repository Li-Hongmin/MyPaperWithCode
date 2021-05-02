function intp = set_sub_k(p)
    %set_sub_k:
    %   version 1.0 --April 2021
    %   Written by Hongmin Li (li.hongmin.xa@alumni.tsukuba.ac.jp)
    %===========
    
    % we want get a list with int number that sum is as same as before

    if numel(p) == 1
        intp = p;
        return
    end

    intp = ceil(p); % intlist for p
    sumright = round(sum(p)); % right sum

    %     if any(intp> ni)
    %         idx = intp > ni;
    %         intp(idx) = ni(idx);
    %     end
    intp(intp < 1) = 1;
    diff = sum(intp) - sumright;

    while diff > 0

        idx = intp > 1;
        u = min(intp(idx));
        idx_u = intp == u;
        n = sum(idx_u);

        if diff < (u -1) * n
            ni = floor(diff / (u - 1));
            %             idx_u = find(idx_u);
            %             intp(idx_u(1:ni)) = 1;
            %             intp(idx_u(ni+1)) = intp(idx_u(ni+1)) - (diff - ni*(u-1));
            % select ni in samples
            randSample = false(1, n);
            randSample(randperm(n, ni)) = true;
            [selectedIdx, restIdx] = deal(idx_u);
            selectedIdx(idx_u) = randSample;
            restIdx(idx_u) = ~randSample;
            % select one in rest idx
            randSample = false(1, n - ni);
            randSample(randperm(n - ni, 1)) = true;
            restIdx(restIdx) = randSample;
            % set selected samples as 1
            intp(selectedIdx) = 1;
            % reduce the supplement
            intp(restIdx) = intp(restIdx) - (diff - ni * (u - 1));

            diff = 0;
        else
            intp(idx_u) = 1;
            diff = diff - (u -1) * n;
        end

    end

    if diff < 0
        print('wrong')
    end

end
