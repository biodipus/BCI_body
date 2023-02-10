%%
files = dir('.\Monkey_data\Neuorns\N\neu*.mat');
edges = [-100,-35,-25,-15,-6,6,15,25,35,100]; bins = length(edges)-1; 
c0 = [repmat(-90,5,1) (1:5)']; c9 = [repmat(90,5,1) (5:9)'];
c1 = [repmat(-45,5,1) (1:5)']; c2 = [repmat(-35,4,1),(2:5)']; c3 = [repmat(-20,3,1),(3:5)'];
c4 = [repmat(-10,2,1) (4:5)']; c5 = [repmat(10,2,1),(5:6)']; c6 = [repmat(20,3,1),(5:7)'];
c7 = [repmat(35,4,1),(5:8)']; c8 = [repmat(45,5,1),(5:9)']; conds = [c1;c2;c3;c4;[0,5];c5;c6;c7;c8]; 
% conds = [c0;conds;c9]; 
conds = [3*ones(size(conds,1),1) conds]; conds = [[1 0 5];[2 0 5];conds]; 
rots = [-45,-35,-20,-10,0,10,20,35,45]'; 
%% Fig 3G 
load Nweight_Hold_bot50_for_plot; nsim=50; % Nweight_Hold_bot50_for_plot
load('VPTselID_N_HW') % VPTselID_N_HW
nid = id3(:,1);
figure; rng(1); 
% clrs = colormap (flipud('RdBu'));
clrs = colormap(flipud(brewermap([],'RdBu'))); 
xx = edges; xx(1)=-45; xx(end) = 45; xx=(xx(2:end)+xx(1:end-1))/2; xx=xx';
for isim = 1:50
    lik1 = nanmean(prob(isim,:,nid,1),3); % lik2 = nanmean(prob(isim,:,nid,2),3);     

    id = (~isnan(lik1) & (conds(:,1)==3)'); con = conds(id,:); lik1 = lik1(id); 
 
    scatter(con(:,2)+2*randn(size(con,1),1),xx(con(:,3))+2*randn(size(con,1),1),10,lik1(1,:),'filled'); hold on;
end
xlabel('Disparity (deg)'); ylabel('Drift (deg)'); 
c=colorbar; c.Label.String = 'VP weight';  caxis([0.4,0.6]);
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); xlim([-50,50]);  
set(gcf,'Position',[1212  389  436  323]);
%% vpweight all neurons
nsim = 500; prob = nan(nsim,size(conds,1),length(files)); prob2 = prob; 
nidc = ones(length(files),2); nidc(:,2)=0;

for f = 1:length(files)
    file = files(f).name; load(file); dat = data.all; info = data.info; 
    if isempty(dat)
        nidc(f,1) = 0; continue;
    end    
    types = info(:,[3,4,6,7]); types(:,3) = ceil(types(:,3)/9); % tpLen = cellfun(@length,dat); 
    rots = unique(types(:,4)); % tgts = unique(types(:,3)); % 1-9T
    
    idx1 = (types(:,2)==6 & types(:,1)==1); idx2 = (types(:,1)==2); 
    idx3 = (types(:,2)==6 & types(:,1)==3); idx4 = (types(:,2)==21 & types(:,1)==3); 
    
    if sum(idx1)==0 || sum(idx2)==0 || sum(idx3)==0 
        nidc(f,1) = 0; continue;
    end
    idx = (idx1 | idx2 | idx3 |idx4); dat = dat(idx); types = types(idx,:); 
    
    idx = (abs(types(:,4))==90); types = types(~idx,:); dat = dat(~idx); rots = unique(types(:,4));   
    if length(rots) ~= 9 || sum(ismember(rots,[-45,-35,-20,-10,0,10,20,35,45])) ~= 9
        nidc(f,1) = 0; continue;
    end

%     if length(rots) ~= 11 || sum(ismember(rots,[-90,-45,-35,-20,-10,0,10,20,35,45,90])) ~= 11
%         nidc(f,1) = 0; continue;
%     end

    if sum(idx4) ~= 0 
        nidc(f,2) = 1;
    end

    FR = []; drf = []; pos = []; 
    for itrl = 1:size(dat,1)
        trial = dat{itrl}; len = size(trial,1); time = trial(:,5); 
        time2 = trial(:,6); idx2 = find(abs(time2)==min(abs(time2)),1); 
        FR(itrl,:) = mean(trial(time>-0.5 & time<0,17),1);        
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
%     idx2 = idx; 
%     for irot = 1:length(rots)
%         idx{irot+2} = (types(:,1)==3 & types(:,4)==rots(irot) & types(:,2)==6);
%         idx2{irot+2} = (types(:,1)==3 & types(:,4)==rots(irot) & types(:,2)==21);
%     end

    x1 = pos(idx{1},1); y1 = FR(idx{1}); 
    x2 = pos(idx{2},1); y2 = FR(idx{2}); 
    [beta,~,J1] = nlinfit(x1,y1,@vonMises,ones(1,3));   
    [beta2,~,J2] = nlinfit(x2,y2,@vonMises,ones(1,3));
    lam1 = vonMises(beta,pos(:,2)); lam2 = vonMises(beta2,pos(:,1)); lam1 = lam1+0.01; lam2 = lam2+0.01;

%     lam1 = nanmean(FR(idx{1})); lam2 = nanmean(FR(idx{2})); lam1 = lam1+0.01; lam2 = lam2+0.01; 
    pr1 = poisspdf(round(FR),lam1); pr2 = poisspdf(round(FR),lam2); pr1 = pr1./(pr1+pr2); pr2 = 1-pr1;
    
    [~,~,ind] = histcounts(drf,edges); ind = [types(:,1),types(:,4),ind]; [~,loc] = ismember(ind,conds,'rows'); % [cond,rot,drf_idx];
    
    % bootstrapping
    for ic = 1:size(conds,1)
        p1 = pr1(loc==ic & types(:,2)~=21); % p2 = pr2(loc==ic & types(:,2)~=21); 
        if isempty(p1)
            continue;
        end
%         prob(1:length(p1),ic,f,1) = p1; 
        id1 = randi(length(p1),nsim,1); prob(:,ic,f,1) = p1(id1); % prob(:,ic,f,2) = p2(id1);
               
        p21 = pr1(loc==ic & types(:,2)~=6); % p22 = pr2(loc==ic & types(:,2)~=6); 
        if isempty(p21)
            continue;
        end
%         prob2(1:length(p21),ic,f,1) = p21;     
        id2 = randi(length(p21),nsim,1); prob2(:,ic,f,1) = p21(id2); % prob2(:,ic,f,2) = p22(id2);  
    end
end

%%
colormap(flipud(brewermap([],'RdBu'))); 
% scatter(conds(:,2),xx(conds(:,3)),50); text(conds(:,2),xx(conds(:,3)),num2str((1:size(conds,1))')); 
lik1 = nanmean(prob(:,:,nid,1),3); lik1 = nanmean(lik1); 
% lik1 = nanmean(log(prob(:,:,nid,1)),3); lik1 = exp(lik1); 
% lik1 = nanmean(lik1); lik1 = (lik1-nanmin(lik1(:)))/(nanmax(lik1(:))-nanmin(lik1(:))); 
scatter(conds(:,2),xx(conds(:,3)),50,lik1,'filled'); caxis([0.4,0.6]);
ylim([-50,50]); xlabel('Disparity (deg)'); ylabel('Drift (deg)'); 
cl = colorbar; cl.Label.String = 'VP weight'; 
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); 
set(gcf,'Position',[1230         566         374         261]); 

