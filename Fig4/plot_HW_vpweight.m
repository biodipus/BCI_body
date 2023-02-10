%%
edges = [-100,-35,-25,-15,-6,6,15,25,35,100]; bins = length(edges)-1; 
c1 = [repmat(-45,5,1) (1:5)']; c2 = [repmat(-35,4,1),(2:5)']; c3 = [repmat(-20,3,1),(3:5)'];
c4 = [repmat(-10,2,1) (4:5)']; c5 = [repmat(10,2,1),(5:6)']; c6 = [repmat(20,3,1),(5:7)'];
c7 = [repmat(35,4,1),(5:8)']; c8 = [repmat(45,5,1),(5:9)']; conds = [c1;c2;c3;c4;[0,5];c5;c6;c7;c8]; 
conds = [3*ones(size(conds,1),1) conds]; conds = [[1 0 5];[2 0 5];conds]; 
rots = [-45,-35,-20,-10,0,10,20,35,45]'; 
%% Fig 4G arm
H = importdata('Hweight_Hold_bot50_for_plot_HW.mat'); 
N = importdata('Nweight_Hold_bot50_for_plot_HW.mat'); 
prob = cat(3, H.prob, N.prob); % prob:hand;
% prob = cat(3, H.prob2, N.prob2); % prob2: wood
nsim=50; % Nweight_Hold_bot50_for_plot

figure; rng(1); 
% colormap(parula); 
clrs = colormap(flipud(brewermap([],'RdBu'))); 
xx = edges; xx(1)=-45; xx(end) = 45; xx=(xx(2:end)+xx(1:end-1))/2; xx=xx';
for isim = 1:50
    lik1 = nanmean(prob(isim,:,:,1),3); % lik2 = nanmean(prob(isim,:,nid,2),3);     

    id = (~isnan(lik1) & (conds(:,1)==3)'); con = conds(id,:); lik1 = lik1(id); 
 
    scatter(con(:,2)+2*randn(size(con,1),1),xx(con(:,3))+2*randn(size(con,1),1),10,lik1(1,:),'filled'); hold on;
end
xlabel('Disparity (deg)'); ylabel('Drift (deg)'); 
c=colorbar; c.Label.String = 'VP weight';  caxis([0.4,0.6]);
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); xlim([-50,50]);  
set(gcf,'Position',[1212  389  436  323]);
%% Fig 4G wood
H = importdata('Hweight_Hold_bot50_for_plot_HW.mat'); 
N = importdata('Nweight_Hold_bot50_for_plot_HW.mat'); 
% prob = cat(3, H.prob, N.prob); % prob:hand;
prob = cat(3, H.prob2, N.prob2); % prob2: wood
nsim=50; % Nweight_Hold_bot50_for_plot

figure; rng(1); 
% colormap(parula); 
clrs = colormap(flipud(brewermap([],'RdBu'))); 
xx = edges; xx(1)=-45; xx(end) = 45; xx=(xx(2:end)+xx(1:end-1))/2; xx=xx';
for isim = 1:50
    lik1 = nanmean(prob(isim,:,:,1),3); % lik2 = nanmean(prob(isim,:,nid,2),3);     

    id = (~isnan(lik1) & (conds(:,1)==3)'); con = conds(id,:); lik1 = lik1(id); 
 
    scatter(con(:,2)+2*randn(size(con,1),1),xx(con(:,3))+2*randn(size(con,1),1),10,lik1(1,:),'filled'); hold on;
end
xlabel('Disparity (deg)'); ylabel('Drift (deg)'); 
c=colorbar; c.Label.String = 'VP weight';  caxis([0.4,0.6]);
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); xlim([-50,50]);  
set(gcf,'Position',[1212  389  436  323]);

%% Fig 4F arm
files = dir('D:\Projects\MI\BCI_data\Monkey_data\Neuorns\N\neu*.mat'); k=0; %E:\Fang\HandRotation\Nico\add_fr\add_fr
xx = -50:50; % pvals = nan(length(files),11); 
% colormap(parula); 
colormap(flipud(brewermap([],'RdBu'))); 
t = linspace(-1,1,160); t2=linspace(-1,1,80); tp=80; 
for f = 159 % Fig 4F example
    file = files(f).name; 
    load(['D:\Projects\MI\BCI_data\Monkey_data\Neuorns\N\',file]);
    dat = data.all; info = data.info; 
    if isempty(dat)
        continue;
    end    
    types = info(:,[3,4,6,7]); types(:,3) = ceil(types(:,3)/9); % tpLen = cellfun(@length,dat); 
    rots = unique(types(:,4)); % tgts = unique(types(:,3)); % 1-9T
    
    if length(rots) ~= 11 || sum(ismember(rots,[-90,-45,-35,-20,-10,0,10,20,35,45,90])) ~= 11
        continue;
    end
    
%     idx = (types(:,2)~=21); dat=dat(idx); types=types(idx,:); 
    idx1 = (types(:,2)==6 & types(:,1)==1); idx2 = (types(:,1)==2); idx3 = (types(:,2)==6 & types(:,1)==3); idx4 = (types(:,2)==21 & types(:,1)==3); 
    if sum(idx1)==0 || sum(idx2)==0 || sum(idx3)==0 
        continue;
    else
        nidc(f,1) = 1;  
    end
    idx = (idx1 | idx2 | idx3 |idx4); dat = dat(idx); types = types(idx,:);   
    
    if sum(idx4) ~= 0 && sum(idx3) ~= 0
        nidc(f,2) = 1;
    end

    
    FR = []; drf = []; pos = []; 
    for itrl = 1:size(dat,1)
        trial = dat{itrl}; len = size(trial,1); 
        time2 = trial(:,6); idx2 = find(abs(time2)==min(abs(time2)),1); 
        FR(itrl,:) = mean(trial(time2>-0.5 & time2<0,17),1);        
        drf(itrl,1) = nanmean(trial(idx2-5:idx2+5,14),1); 
        pos(itrl,:) = nanmean(trial(idx2-5:idx2+5,15:16),1); 
    end
    
    idc = []; 
    for irot = 1:length(rots)
        idx = find(types(:,4)==rots(irot)); d = drf(idx); id = find(delOutliers(d)); 
        idc = [idc;idx(id)]; 
    end
    drf(idc) = nan; idx = (~isnan(drf)); 
    FR=FR(idx,:); types=types(idx,:); drf=drf(idx,:); pos=pos(idx,:);
    
    idx = {}; idx{1} = (types(:,1)==1 & types(:,2)==6); idx{2} = (types(:,1)==2); 
    idx2 = idx; 
    for irot = 1:length(rots)
        idx{irot+2} = (types(:,1)==3 & types(:,4)==rots(irot) & types(:,2)==6);
        idx2{irot+2} = (types(:,1)==3 & types(:,4)==rots(irot) & types(:,2)==21);
    end

    x1 = pos(idx{1},1); y1 = FR(idx{1}); 
    x2 = pos(idx{2},1); y2 = FR(idx{2}); 
    [beta,~,J1] = nlinfit(x1,y1,@vonMises,ones(1,3));   
    [beta2,~,J2] = nlinfit(x2,y2,@vonMises,ones(1,3));

    lam1 = vonMises(beta,pos(:,2)); lam2 = vonMises(beta2,pos(:,1)); lam1 = lam1+0.01; lam2 = lam2+0.01;
    pr1 = poisspdf(round(FR),lam1); pr2 = poisspdf(round(FR),lam2); pr1 = pr1./(pr1+pr2); pr2 = 1-pr1;
%     est = pr1.*pos(:,1) + pr2.*pos(:,2); estd = est-pos(:,1); 
    
%     [~,~,ind] = histcounts(drf,edges); ind = [types(:,1),types(:,4),ind]; [~,loc] = ismember(ind,conds,'rows'); % [cond,rot,drf_idx];

    plot([-55,55],[0,0],'k','linewidth',2); hold on; plot(-55:55,-55:55,'Color',0.7*ones(1,3),'linewidth',2); siz = 50; jit=1.2; 
    cond3 = (types(:,1)==3 & types(:,2)==6); % hand
%     cond3 = (types(:,1)==3 & types(:,2)==6); % wood
    cond3 = [types(cond3,4),drf(cond3),pr1(cond3)]; %[rot,drf,pr1];
%     scatter(cond3(:,1)+jit*randn(size(cond3,1),1),cond3(:,2),siz,cond3(:,3),'filled'); % ,'MarkerEdgeColor','w'
    idx = (abs(cond3(:,1))>30 & abs(cond3(:,2))<15); grp1 = cond3(~idx,:); grp2 = cond3(idx,:); 
    grp2 = sortrows(grp2,3); grp2 = flipud(grp2);  
    scatter(grp1(:,1)+1.2*randn(size(grp1,1),1),grp1(:,2),siz,grp1(:,3),'filled','MarkerEdgeColor','w'); hold on;
    scatter(grp2(:,1)+1.2*randn(size(grp2,1),1),grp2(:,2),siz,grp2(:,3),'filled','MarkerEdgeColor','w');
    idx = find(cond3(:,1)==0 & cond3(:,3)>0.7); idx=idx(1:2); scatter(cond3(idx,1)+randn(2,1),cond3(idx,2),siz,cond3(idx,3),'filled');
    xlabel('Disparity (deg)'); ylabel('Drift (deg)'); c=colorbar; c.Label.String = 'VP weight';   
    box off;  xlim([-55,55]); ylim([-55,55]); % view([21,3]); grid off; 
    set(gca,'FontName','Calibri','FontSize',18,'FontWeight','bold'); 
    set(gcf,'Position',[1212  389  436  323]);
%     set(gcf,'Position',[1192         452         344         252]); 
%     set(gcf,'Position',[1192         415         380         289]); % alpha(0.5); 
%     set(gcf,'Position',[1212  389  436  323]);
%     title('Data from large conflict'); xlabel('Drift'); ylabel('VP weight'); 
end
%% Fig 4F wood
files = dir('D:\Projects\MI\BCI_data\Monkey_data\Neuorns\N\neu*.mat'); k=0; %E:\Fang\HandRotation\Nico\add_fr\add_fr
xx = -50:50; % pvals = nan(length(files),11); 
% colormap(parula); 
colormap(flipud(brewermap([],'RdBu'))); 
t = linspace(-1,1,160); t2=linspace(-1,1,80); tp=80; 
for f = 159 % Fig 4F example
    file = files(f).name; 
    load(['D:\Projects\MI\BCI_data\Monkey_data\Neuorns\N\',file]);
    dat = data.all; info = data.info; 
    if isempty(dat)
        continue;
    end    
    types = info(:,[3,4,6,7]); types(:,3) = ceil(types(:,3)/9); % tpLen = cellfun(@length,dat); 
    rots = unique(types(:,4)); % tgts = unique(types(:,3)); % 1-9T
    
    if length(rots) ~= 11 || sum(ismember(rots,[-90,-45,-35,-20,-10,0,10,20,35,45,90])) ~= 11
        continue;
    end
    
%     idx = (types(:,2)~=21); dat=dat(idx); types=types(idx,:); 
    idx1 = (types(:,2)==6 & types(:,1)==1); idx2 = (types(:,1)==2); idx3 = (types(:,2)==6 & types(:,1)==3); idx4 = (types(:,2)==21 & types(:,1)==3); 
    if sum(idx1)==0 || sum(idx2)==0 || sum(idx3)==0 
        continue;
    else
        nidc(f,1) = 1;  
    end
    idx = (idx1 | idx2 | idx3 |idx4); dat = dat(idx); types = types(idx,:);   
    
    if sum(idx4) ~= 0 && sum(idx3) ~= 0
        nidc(f,2) = 1;
    end

    
    FR = []; drf = []; pos = []; 
    for itrl = 1:size(dat,1)
        trial = dat{itrl}; len = size(trial,1); 
        time2 = trial(:,6); idx2 = find(abs(time2)==min(abs(time2)),1); 
        FR(itrl,:) = mean(trial(time2>-0.5 & time2<0,17),1);        
        drf(itrl,1) = nanmean(trial(idx2-5:idx2+5,14),1); 
        pos(itrl,:) = nanmean(trial(idx2-5:idx2+5,15:16),1); 
    end
    
    idc = []; 
    for irot = 1:length(rots)
        idx = find(types(:,4)==rots(irot)); d = drf(idx); id = find(delOutliers(d)); 
        idc = [idc;idx(id)]; 
    end
    drf(idc) = nan; idx = (~isnan(drf)); 
    FR=FR(idx,:); types=types(idx,:); drf=drf(idx,:); pos=pos(idx,:);
    
    idx = {}; idx{1} = (types(:,1)==1 & types(:,2)==6); idx{2} = (types(:,1)==2); 
    idx2 = idx; 
    for irot = 1:length(rots)
        idx{irot+2} = (types(:,1)==3 & types(:,4)==rots(irot) & types(:,2)==6);
        idx2{irot+2} = (types(:,1)==3 & types(:,4)==rots(irot) & types(:,2)==21);
    end

    x1 = pos(idx{1},1); y1 = FR(idx{1}); 
    x2 = pos(idx{2},1); y2 = FR(idx{2}); 
    [beta,~,J1] = nlinfit(x1,y1,@vonMises,ones(1,3));   
    [beta2,~,J2] = nlinfit(x2,y2,@vonMises,ones(1,3));

    lam1 = vonMises(beta,pos(:,2)); lam2 = vonMises(beta2,pos(:,1)); lam1 = lam1+0.01; lam2 = lam2+0.01;
    pr1 = poisspdf(round(FR),lam1); pr2 = poisspdf(round(FR),lam2); pr1 = pr1./(pr1+pr2); pr2 = 1-pr1;
%     est = pr1.*pos(:,1) + pr2.*pos(:,2); estd = est-pos(:,1); 
    
%     [~,~,ind] = histcounts(drf,edges); ind = [types(:,1),types(:,4),ind]; [~,loc] = ismember(ind,conds,'rows'); % [cond,rot,drf_idx];

    plot([-55,55],[0,0],'k','linewidth',2); hold on; plot(-55:55,-55:55,'Color',0.7*ones(1,3),'linewidth',2); siz = 50; jit=1.2; 
%     cond3 = (types(:,1)==3 & types(:,2)==6); % hand
    cond3 = (types(:,1)==3 & types(:,2)==21); % wood
    cond3 = [types(cond3,4),drf(cond3),pr1(cond3)]; %[rot,drf,pr1];
%     scatter(cond3(:,1)+jit*randn(size(cond3,1),1),cond3(:,2),siz,cond3(:,3),'filled'); % ,'MarkerEdgeColor','w'
    idx = (abs(cond3(:,1))>30 & abs(cond3(:,2))<15); grp1 = cond3(~idx,:); grp2 = cond3(idx,:); 
    grp2 = sortrows(grp2,3); grp2 = flipud(grp2);  
    scatter(grp1(:,1)+1.2*randn(size(grp1,1),1),grp1(:,2),siz,grp1(:,3),'filled','MarkerEdgeColor','w'); hold on;
    scatter(grp2(:,1)+1.2*randn(size(grp2,1),1),grp2(:,2),siz,grp2(:,3),'filled','MarkerEdgeColor','w');
    idx = find(cond3(:,1)==0 & cond3(:,3)>0.7); idx=idx(1:2); scatter(cond3(idx,1)+randn(2,1),cond3(idx,2),siz,cond3(idx,3),'filled');
    xlabel('Disparity (deg)'); ylabel('Drift (deg)'); c=colorbar; c.Label.String = 'VP weight';   
    box off;  xlim([-55,55]); ylim([-55,55]); % view([21,3]); grid off; 
    set(gca,'FontName','Calibri','FontSize',18,'FontWeight','bold'); 
    set(gcf,'Position',[1212  389  436  323]);
%     set(gcf,'Position',[1192         452         344         252]); 
%     set(gcf,'Position',[1192         415         380         289]); % alpha(0.5); 
%     set(gcf,'Position',[1212  389  436  323]);
%     title('Data from large conflict'); xlabel('Drift'); ylabel('VP weight'); 
end
%% Fig 4H
N = importdata('Nweight_Hold_HW.mat'); 
prob =  N.prob(:,:,23,:);
prob2 = N.prob2(:,:,23,:);
% probmean = squeeze(nanmean(prob(:,:,:,1),1));

lik1 = prob(:,:,:,1); ml = nan(length(rots),2); 
for irot = 1:length(rots)
    ll = lik1(:,conds(:,1)==3 & conds(:,2)==rots(irot)); 
    ml(irot,1) = nanmean(ll(:));
    num = sum(~isnan(ll(:)));
    ml(irot,2) = nanstd(ll(:))/sqrt(num); 
end
figure; 
% plot(c,lik1,'k.'); 
hold on; 
errorbar(rots,ml(:,1),ml(:,2),'linewidth',2); 

lik1 = prob2(:,:,:,1); ml = nan(length(rots),2); 
for irot = 1:length(rots)
    ll = lik1(:,conds(:,1)==3 & conds(:,2)==rots(irot)); 
    ml(irot,1) = nanmean(ll(:)); 
    num = sum(~isnan(ll(:)));
    ml(irot,2) = nanstd(ll(:))/sqrt(num); 
end
errorbar(rots,ml(:,1),ml(:,2),'linewidth',2); 

xlabel('Disparity (deg)'); ylabel('VP weight'); box off;
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold');
set(gca,'FontName','Calibri','FontWeight','bold','FontSize',14); 
set(gcf,'Position',[702   669   398   299]);
%% Fig 4I
H = importdata('Hweight_Hold_HW.mat'); 
N = importdata('Nweight_Hold_HW.mat'); 
prob = cat(3, H.prob, N.prob);
prob2 = cat(3, H.prob2, N.prob2);
% probmean = squeeze(nanmean(prob(:,:,:,1),1));

lik1 = squeeze(nanmean(prob(:,:,:,1),1)); ml = nan(length(rots),2);  
for irot = 1:length(rots)
    ll = nanmean(lik1(conds(:,1)==3 & conds(:,2)==rots(irot),:),1); 
    ml(irot,1) = nanmean(ll(:)); ml(irot,2) = nanstd(ll(:))/sqrt(length(ll)); 
end
figure; 
% plot(c,lik1,'k.'); 
hold on; 
errorbar(rots,ml(:,1),ml(:,2),'linewidth',2); 

lik1 = squeeze(nanmean(prob2(:,:,:,1),1)); ml = nan(length(rots),2); 
for irot = 1:length(rots)
    ll = nanmean(lik1(conds(:,1)==3 & conds(:,2)==rots(irot),:),1); 
    ml(irot,1) = nanmean(ll(:)); ml(irot,2) = nanstd(ll(:))/sqrt(length(ll)); 
end
% plot(c,lik1,'k.'); hold on; 
errorbar(rots,ml(:,1),ml(:,2),'linewidth',2); 

xlabel('Disparity (deg)'); ylabel('VP weight'); box off;
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold');
set(gca,'FontName','Calibri','FontWeight','bold','FontSize',14); 
set(gcf,'Position',[702   669   398   299]);