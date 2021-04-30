function labels = script_dnc_sc(fea, k)
    if (size(fea,1) > 1000000)
        u = 50;
    else
        u = 200;
    end
    labels = DnC_SC(fea, k, p, u, knn);
end