%% Fig S8
edges = [-100,-35,-25,-15,-6,6,15,25,35,100]; bins = length(edges)-1; 
c0 = [repmat(-90,5,1) ([2:5,5])']; c9 = [repmat(90,5,1) ([5:7,7,7])'];
c1 = [repmat(-45,5,1) (1:5)']; c2 = [repmat(-35,4,1),(2:5)']; c3 = [repmat(-20,3,1),(3:5)'];
c4 = [repmat(-10,2,1) (4:5)']; c5 = [repmat(10,2,1),(5:6)']; c6 = [repmat(20,3,1),(5:7)'];
c7 = [repmat(35,4,1),(5:8)']; c8 = [repmat(45,5,1),(5:9)']; conds = [c0;c1;c2;c3;c4;[0,5];c5;c6;c7;c8;c9]; 
conds = [3*ones(size(conds,1),1) conds]; conds = [[1 0 5];[2 0 5];conds]; 
rots = [-90,-45,-35,-20,-10,0,10,20,35,45,90]'; 
H = importdata('vpweight_with_rot90_H.mat'); 
N = importdata('vpweight_with_rot90_N.mat'); 
H.prob = H.prob(:,:,:,1);
N.prob = N.prob(:,:,:,3);
prob = cat(3, H.prob, N.prob); 
nsim=50; 

figure; rng(1); 
% colormap(parula); 
clrs = colormap(flipud(brewermap([],'RdBu'))); 
xx = edges; 
xx(1)=-45; xx(end) = 45;
xx=(xx(2:end)+xx(1:end-1))/2; xx=xx';
for isim = 1:50
    lik1 = nanmean(prob(isim,:,:),3);    

    id = (~isnan(lik1) & (conds(:,1)==3)'); con = conds(id,:); lik1 = lik1(id); 
 
    scatter(con(:,2)+2*randn(size(con,1),1),xx(con(:,3))+2*randn(size(con,1),1),10,lik1(1,:),'filled'); hold on;
end
xlabel('Disparity (deg)'); ylabel('Drift (deg)'); 
c=colorbar; c.Label.String = 'VP weight'; 
caxis([0.3,0.65]);
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); 
% xlim([-50,50]);  
set(gcf,'Position',[1212  389  436  323]);
%%
figure()
lik1 = squeeze(nanmean(prob(:,:,:),1)); ml = nan(length(rots),2);  
for irot = 1:length(rots)
    ll = nanmean(lik1(conds(:,1)==3 & conds(:,2)==rots(irot),:),1); 
    ml(irot,1) = nanmean(ll(:)); ml(irot,2) = nanstd(ll(:))/sqrt(length(ll)); 
end
figure; 

hold on; 
errorbar(rots,ml(:,1),ml(:,2),'linewidth',2); 

xlabel('Disparity (deg)'); ylabel('VP weight'); box off;
set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold');
set(gca,'FontName','Calibri','FontWeight','bold','FontSize',14); 
set(gcf,'Position',[702   669   398   299]);