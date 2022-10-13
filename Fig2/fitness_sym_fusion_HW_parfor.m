function [ y ] = fitness_update(params,realdata_h,realdata_w,rots,c)
%% Predict result
rotation = rots';
%????????
sp = params(1); sv = params(2); sigma_prior = 10000; %pc_h = params(3)/10; pc_w = params(4)/10; %sigma_prior = params(3)*10;

numsimu = 5000;
sp=repmat(sp,numsimu,length(rotation));
sv=repmat(sv,numsimu,length(rotation));
rotation=repmat(rotation,numsimu,1);
mu_prior = rotation./2;

x1 = random('Normal',0,sp);
x2 = random('Normal',rotation,sv);
estp = [];
est1 = (x1./sp.^2+x2./sv.^2+mu_prior/sigma_prior^2)./(1./sp.^2+1./sv.^2+1/sigma_prior^2);
% est2p = (x1./sp.^2+mu_prior/sigma_prior^2)./(1./sp.^2+1/sigma_prior^2);
% est2v = (x2./sv.^2+mu_prior/sigma_prior^2)./(1./sv.^2+1/sigma_prior^2);
% sigma = sqrt(sv.^2.*sp.^2+sv.^2*sigma_prior^2+sp.^2*sigma_prior^2);
% pa1 = ((x2-x1).^2*sigma_prior^2+(x2-mu_prior).^2.*sp.^2+(x1-mu_prior).^2.*sv.^2)./sigma.^2;
% likec = 1./(2*pi*sigma).*exp(-1/2*pa1);
% pa2 = (x2-mu_prior).^2./(sv.^2+sigma_prior^2) + (x1-mu_prior).^2./(sp.^2+sigma_prior^2);
% liked = 1./(2*pi*sqrt((sv.^2+sigma_prior^2).*(sp.^2+sigma_prior^2))).*exp(-1/2*pa2);
% postc = likec*pc_h./(likec*pc_h+liked*(1-pc_h));
% % postd = liked*(1-pc_h)./(likec*pc_h+liked*(1-pc_h));
% if c==2 %| c==4
%     %%%%%%%%% select
%     epsilon = 0.5;
%     estp(postc<epsilon) = x1(postc<epsilon);
%     estp(postc>=epsilon) = est1(postc>=epsilon);
%     estp = reshape(estp,numsimu,size(rotation,2));
% elseif c==3 %| c ==6
%     %%%%%%%%% Matching
%     epsilon = random('uniform',0,1,numsimu,length(rots));
%     estp(postc<epsilon) = x1(postc<epsilon);
%     estp(postc>=epsilon) = est1(postc>=epsilon);
%     estp = reshape(estp,numsimu,size(rotation,2));
% elseif c==1 %| c==2
%     %%%%%% average
%     estp = postc.*est1 + (1-postc).*est2p;
% end
predict{1} = est1;
%% wood
if ~isempty(realdata_w)
    x1 = random('Normal',0,sp);
    x2 = random('Normal',rotation,sv);
    estp = [];
    est1 = (x1./sp.^2+x2./sv.^2+mu_prior/sigma_prior^2)./(1./sp.^2+1./sv.^2+1/sigma_prior^2);
%     est2p = (x1./sp.^2+mu_prior/sigma_prior^2)./(1./sp.^2+1/sigma_prior^2);
%     est2v = (x2./sv.^2+mu_prior/sigma_prior^2)./(1./sv.^2+1/sigma_prior^2);
%     sigma = sqrt(sv.^2.*sp.^2+sv.^2*sigma_prior^2+sp.^2*sigma_prior^2);
%     pa1 = ((x2-x1).^2*sigma_prior^2+(x2-mu_prior).^2.*sp.^2+(x1-mu_prior).^2.*sv.^2)./sigma.^2;
%     likec = 1./(2*pi*sigma).*exp(-1/2*pa1);
%     pa2 = (x2-mu_prior).^2./(sv.^2+sigma_prior^2) + (x1-mu_prior).^2./(sp.^2+sigma_prior^2);
%     liked = 1./(2*pi*sqrt((sv.^2+sigma_prior^2).*(sp.^2+sigma_prior^2))).*exp(-1/2*pa2);
%     postc = likec*pc_w./(likec*pc_w+liked*(1-pc_w));
%     % postd = liked*(1-pc_w)./(likec*pc_w+liked*(1-pc_w));
%     if c==2 %| c==4
%         %%%%%%%%% select
%         epsilon = 0.5;
%         estp(postc<epsilon) = x1(postc<epsilon);
%         estp(postc>=epsilon) = est1(postc>=epsilon);
%         estp = reshape(estp,numsimu,size(rotation,2));
%     elseif c==3 %| c ==6
%         %%%%%%%%% Matching
%         epsilon = random('uniform',0,1,numsimu,length(rots));
%         estp(postc<epsilon) = x1(postc<epsilon);
%         estp(postc>=epsilon) = est1(postc>=epsilon);
%         estp = reshape(estp,numsimu,size(rotation,2));
%     elseif c==1 %| c==2
%         %%%%%% average
%         estp = postc.*est1 + (1-postc).*est2p;
%     end
    predict{2} = est1;
end
%% Realistic result
%figure;
%for i = 1:length(est)
%    plot(rotation,-est(i,:),'.'); hold on
%end
%errorbar(rotation,-mean(est),std(est));
% global reach;
% if isempty(reach)
%     load humExp;
% end
% dat = reach(reach(:,1)==1,:);   % choose one out of 8 subjects
% cond3 = dat(dat(:,2)==3,4:5); rots = unique(cond3(:,1)); drf = nan(length(rots),100); drfsd=drf;
% for i = 1:length(rots)
%     %plot(cond3(cond3(:,1)==rots(i),1),cond3(cond3(:,1)==rots(i),2),'b.'); hold on;
%     aa = cond3(cond3(:,1)==rots(i),2);
%     drf(i,1:length(aa)) = aa;
%     %     drfsd(i,1) = nanstd(cond3(cond3(:,1)==rots(i),2));
% end
% realistic = -drf';

%%
% realistic = realdata;
% rots = rots;
% if 1 %mod(c,2)==1

for j = 1:length(rots) %floor(length(rots)/2)+1
    pd = fitdist(predict{1}(:,j),'Kernel');
    xreal = realdata_h(~isnan(realdata_h(:,j)),j);
    ys = pdf(pd,xreal);
    cost{1}(j) = nansum(log(ys(~isnan(ys))));
end
if ~isempty(realdata_w)
    for j = 1:length(rots) %floor(length(rots)/2)+1
        pd = fitdist(predict{2}(:,j),'Kernel');
        xreal = realdata_w(~isnan(realdata_w(:,j)),j);
        ys = pdf(pd,xreal);
        cost{2}(j) = nansum(log(ys(~isnan(ys))));
    end
else
    cost{2} = 0;
end

y = -(nansum(cost{1}) + nansum(cost{2}));
end
