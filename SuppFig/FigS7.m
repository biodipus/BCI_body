%%
files = dir('.\Monkey_data\Neuorns\H\neu*.mat');
edges = [-100,-35,-25,-15,-6,6,15,25,35,100]; bins = length(edges)-1; 
xx = edges; xx(1)=-45; xx(end) = 45; xx=(xx(2:end)+xx(1:end-1))/2; xx=xx';
c1 = [repmat(-45,5,1) (1:5)']; c2 = [repmat(-35,4,1),(2:5)']; c3 = [repmat(-20,3,1),(3:5)'];
c4 = [repmat(-10,2,1) (4:5)']; c5 = [repmat(10,2,1),(5:6)']; c6 = [repmat(20,3,1),(5:7)'];
c7 = [repmat(35,4,1),(5:8)']; c8 = [repmat(45,5,1),(5:9)']; conds = [c1;c2;c3;c4;[0,5];c5;c6;c7;c8]; 
conds = [3*ones(size(conds,1),1) conds]; conds = [[1 0 5];[2 0 5];conds]; 
rots = [-45,-35,-20,-10,0,10,20,35,45]';
%% Fig S7A H
load Hweight_Ref_bot50_for_plot; nsim=50; 
figure; rng(1); 
% clrs = colormap (flipud('RdBu'));
clrs = colormap(flipud(brewermap([],'RdBu'))); 
xx = edges; xx(1)=-45; xx(end) = 45; xx=(xx(2:end)+xx(1:end-1))/2; xx=xx';
lik1_all = [];
for isim = 1:50
    lik1 = nanmean(prob(isim,:,:,1),3); % lik2 = nanmean(prob(isim,:,nid,2),3);     

    id = (~isnan(lik1) & (conds(:,1)==3)'); con = conds(id,:); lik1 = lik1(id); 
 
    scatter(con(:,2)+2*randn(size(con,1),1),xx(con(:,3))+2*randn(size(con,1),1),10,lik1(1,:),'filled'); hold on;

    lik1_all = [lik1_all;lik1];
end
xlabel('Disparity (deg)'); ylabel('Drift (deg)'); 
c=colorbar; c.Label.String = 'VP weight';  caxis([0.4,0.6]);
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); xlim([-50,50]);  
set(gcf,'Position',[1212  389  436  323]);

figure();
% lik1_err = std(lik1_all);
% lik1_all = mean(lik1_all);
likmean = [];
likerrmean = [];
for i = 1:length(rots)
    idx = con(:,2) == rots(i);
    tmp = reshape(lik1_all(:,idx),[],1);
    likmean(i) = mean(tmp);
    likerrmean(i) = std(tmp);
end 
e = errorbar(rots,likmean,likerrmean,'LineWidth',2);
e.CapSize = 0;
axis padded
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); xlim([-50,50]);  
set(gcf,'Position',[1212  389  436  323]);
ylim([0.4,0.6])
%% Fig S7A N
load Nweight_Ref_bot50_for_plot; nsim=50; 
figure; rng(1); 
% clrs = colormap (flipud('RdBu'));
clrs = colormap(flipud(brewermap([],'RdBu'))); 
xx = edges; xx(1)=-45; xx(end) = 45; xx=(xx(2:end)+xx(1:end-1))/2; xx=xx';
lik1_all = [];
for isim = 1:50
    lik1 = nanmean(prob(isim,:,:,1),3); % lik2 = nanmean(prob(isim,:,nid,2),3);     

    id = (~isnan(lik1) & (conds(:,1)==3)'); con = conds(id,:); lik1 = lik1(id); 
 
    scatter(con(:,2)+2*randn(size(con,1),1),xx(con(:,3))+2*randn(size(con,1),1),10,lik1(1,:),'filled'); hold on;

    lik1_all = [lik1_all;lik1];
end
xlabel('Disparity (deg)'); ylabel('Drift (deg)'); 
c=colorbar; c.Label.String = 'VP weight';  caxis([0.4,0.6]);
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); xlim([-50,50]);  
set(gcf,'Position',[1212  389  436  323]);


figure();
% lik1_err = std(lik1_all);
% lik1_all = mean(lik1_all);
likmean = [];
likerrmean = [];
for i = 1:length(rots)
    idx = con(:,2) == rots(i);
    tmp = reshape(lik1_all(:,idx),[],1);
    likmean(i) = mean(tmp);
    likerrmean(i) = std(tmp);
end 
e = errorbar(rots,likmean,likerrmean,'LineWidth',2);
e.CapSize = 0;
axis padded
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); xlim([-50,50]);  
set(gcf,'Position',[1212  389  436  323]);
ylim([0.4,0.6])