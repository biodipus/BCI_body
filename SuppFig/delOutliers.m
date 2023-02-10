function [ outliers ] = delOutliers( x )

% outliers = abs(x-nanmedian(x))>3*nanstd(x);

rng = iqr(x); q1 = quantile(x,0.25); q3 = quantile(x,0.75);
outliers = x < (q1-1.5*rng) | x > (q3+1.5*rng);

end

