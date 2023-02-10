function [ d ] = vonMises( params, theta )

% alpha = params(1); beta = params(2); mu = params(3); b = params(4);
% d = b + alpha*exp(beta*cos(theta-mu));
theta = deg2rad(theta);
b = params(1); alpha = params(2); mu = params(3); % beta = params(4); % sigma = params(4); 
% if length(theta)>3
%     w = params(4); 
% else
%     w = 1;
% end
% w = params(4);
d = b + alpha*cos(theta-mu);

d(d<0) = 0; 

% d = b + alpha*cos(0.01*(theta-mu)); 
% d = b + alpha*cos(beta*(theta-mu));
% d = b + alpha*exp(beta*cos(theta-mu));

% d = b + alpha*exp(beta*(cos(theta-mu)-1));

end

