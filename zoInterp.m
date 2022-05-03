function x_interp = zoInterp(x, numInterp)
    x_interp = reshape(repmat(x,numInterp,1),1,length(x)*numInterp);
end

