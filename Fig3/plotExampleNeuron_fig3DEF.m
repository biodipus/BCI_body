%% Fig 3D F 
files = dir('D:\Projects\MI\BCI_data\Monkey_data\Neuorns\H\neu*.mat'); k=0; xx = -50:50; 
drawArrow = @(xy,c) quiver(xy(1,1),xy(1,2),xy(2,1)-xy(1,1), xy(2,2)-xy(1,2),'MaxHeadSize',0.2,'Color',c,'LineWidth',2); 
t = linspace(-1,1,160); t2=linspace(-1,1,80); tp=80; % oxy = [421,296.25];
for f = 195
    file = files(f).name; 
    load(['D:\Projects\MI\BCI_data\Monkey_data\Neuorns\H\',file]);  
    dat = data.all; info = data.info; 
    if isempty(dat)
        k=k+1; continue;
    end    
    type = data.type; 
    info1 = info; info1(:,6) = ceil(info1(:,6)/9); types = info1(:,[3,4,6,7]); 
    types(:,5) = type(:,8); 
    rots = unique(types(:,4)); 
    
    idx = (types(:,2)~=21); dat=dat(idx); types=types(idx,:); spk = data.spks;
    if isempty(dat) || sum(types(:,1)==1)==0 || sum(types(:,1)==2)==0 || sum(types(:,1)==3)==0
        k=k+1; continue;
    end
    
    if length(rots) ~= 9
        k=k+1;continue;
    end
    
    FR = []; drf = []; pos = []; 
    for itrl = 1:size(dat,1)
        trial = dat{itrl}; len = size(trial,1);
        time2 = trial(:,3); idx2 = find(abs(time2)==min(abs(time2)),1);
        FR(itrl,:) = mean(trial(time2>-0.5 & time2<0,10),1);
        drf(itrl,1) = nanmean(trial(idx2-5:idx2+5,7),1);
        pos(itrl,:) = nanmean(trial(idx2-5:idx2+5,8:9),1);
    end
    drf = -drf; 
    
    idc = []; 
    for irot = 1:length(rots)
        idx = find(types(:,4)==rots(irot)); d = drf(idx); id = find(delOutliers(d)); 
        idc = [idc;idx(id)]; 
    end
    drf(idc) = nan; idx = (~isnan(drf)); 
    FR=FR(idx,:); types=types(idx,:); drf=drf(idx,:); pos=pos(idx,:); type=type(idx,:); 
    
    idx = {}; idx{1} = (types(:,1)==1 & types(:,2)==6); idx{2} = (types(:,1)==2); 
%     for irot = 1:length(rots)
%         idx{irot+2} = (types(:,1)==3 & types(:,2)==6 & types(:,4)==rots(irot));
%     end

    x1 = pos(idx{1},1); y1 = FR(idx{1}); 
    [beta,~,J1] = nlinfit(x1,y1,@vonMises,ones(1,3)); 
    
    x2 = pos(idx{2},1); y2 = FR(idx{2}); 
    [beta2,~,J2] = nlinfit(x2,y2,@vonMises,ones(1,3));

    lam1 = vonMises(beta,pos(:,2)); lam2 = vonMises(beta2,pos(:,1)); lam1 = lam1+0.01; lam2 = lam2+0.01;
    pr1 = poisspdf(round(FR),lam1); pr2 = poisspdf(round(FR),lam2); pr1 = pr1./(pr1+pr2); pr2 = 1-pr1;
    idx3 = (types(:,1)==3 & types(:,2)==6); cond3 = [types(idx3,4:5),drf(idx3),pos(idx3,:),pr1(idx3)]; 
    type3 = type(idx3,:); fr3 = FR(idx3); 

%     scatter(cond3(:,1)+2*randn(size(cond3,1),1),cond3(:,3),50,cond3(:,4),'filled','MarkerEdgeColor','w'); 

%     figure; plot(cond3(:,1),cond3(:,3),'ko','markerfacecolor','k','markersize',3); hold on; 
%     plot([-50,50],[0,0],'k','linewidth',2); 
%     plot(-50:50,-50:50,'Color',0.7*ones(1,3),'linewidth',2); box off;
%     set(gca,'XTick',-40:20:40,'YTick',-40:20:40); set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); 
%     set(gcf,'Position',[1192         415         335         289]); 
%     xlabel('Disparity (deg)'); ylabel('Drift (deg)'); 
    
%     idx1 = find(cond3(:,1)==35 & abs(cond3(:,6)-0.6)<0.1); idx1=idx1(2); 
    idx1 = 12;
    hp = cond3(idx1,4); vp = cond3(idx1,5); 
%     clrs = colormap(flipud(brewermap([],'RdBu'))); clr = clrs(round(linspace(1,64,6)),:); 
%     plot(cond3(idx1,1),cond3(idx1,3),'o','markersize',3,'markerfacecolor','r'); 
    
    figure; clrs = colormap(parula);%clrs = colormap(flipud(brewermap([],'RdBu'))); 
    clrs = clrs(round(linspace(1,256,length(rots))),:); clrs = [0.7*ones(1,3); 0.3*ones(1,3); clrs];
    beta = []; pref = []; 
    for ic = 1:2
        hpos = pos(idx{ic},1); fr = FR(idx{ic}); [~,id] =max(fr); pref(ic,1) = hpos(id); 
        b = nlinfit(hpos,fr,@vonMises,ones(1,3)); yy = vonMises(b,xx);
        if ic == 1
            h1 = plot(xx,yy,'Color',clrs(ic,:),'linewidth',2); hold on; 
            yc1 = vonMises(b,vp); 
            scatter(vp,yc1,30,0.7*ones(1,3),'filled'); 
        elseif ic == 2
            h2 = plot(xx,yy,'Color',clrs(ic,:),'linewidth',2); 
            yc2 = vonMises(b,hp); scatter(hp,yc2,30,0.3*ones(1,3),'filled'); 
        end   
    end
    nam = ['\itf_{VP}\rm(\vartheta)';'\itf_{P}\rm(\vartheta)';cellstr([num2str(rots),repmat('бу',9,1)])]; 
    h = legend([h1,h2],nam(1:2),'Location','northeast'); set(h,'box','off');
    box off; xlabel('Hand position (deg)'); ylabel('Firing Rate (spikes/s)'); 
    set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); xlim([-70,70]); ylim([0,8]);
    set(gcf,'Position',[1192         415         380         289]); 
    
    plot(cond3(idx1,2),fr3(idx1),'o','Color',clrs(8,:),'MarkerFaceColor',clrs(8,:)); 
    plot(cond3(idx1,2),8,'r+','linewidth',2); % weight = 0.6
%     plot(cond3(idx1,2),fr3(idx1),'o','Color',clrs(4,:),'MarkerFaceColor',clrs(4,:)); plot(cond3(idx1,2),8,'r+','linewidth',2); % weight = 0.3
    plot([cond3(idx1,2),cond3(idx1,2)],[0,8],'k--'); 
    hvec = [hp,yc2]; vvec = [vp,yc1]; w = cond3(idx1,6); tvec = [cond3(idx1,2),fr3(idx1)]; 
%     drawArrow([0 0;hvec],clrs(3,:)); drawArrow([0,0;vvec],clrs(11,:)); d=drawArrow([0,0;tvec],clrs(8,:)); d.LineStyle = '--';
%     h1=arrow([0,0],hvec); arrow(h1,'facecolor',clrs(3,:),'edgecolor',clrs(3,:),'Tipangle',10,'Length',7); 
%     h2=arrow([0,0],vvec); arrow(h2,'facecolor',clrs(11,:),'edgecolor',clrs(11,:),'Tipangle',10,'Length',7);
%     h3=arrow([0,0],tvec); arrow(h3,'facecolor',clrs(8,:),'edgecolor',clrs(8,:),'Tipangle',10,'Length',7); % weight = 0.6
%     h3=arrow([0,0],tvec); arrow(h3,'facecolor',clrs(4,:),'edgecolor',clrs(4,:),'Tipangle',10,'Length',7); % weight = 0.3
%     plot([0,hvec(1)]',[0,hvec(2)]','Color',clrs(3,:),'linewidth',2,'linestyle','--'); 
%     plot([0,vvec(1)]',[0,vvec(2)]','Color',clrs(11,:),'linewidth',2,'linestyle','--'); 
%     plot([0,tvec(1)]',[0,tvec(2)]','Color',clrs(8,:),'linewidth',2); 
%     title('VP weight = 0.6'); 
    
    x = 0:20; y1 = poisspdf(x,yc1); y2 = poisspdf(x,yc2); 
    ax1 = axes('Position',[0.6,0.33,0.2,0.2]); % weight = 0.6
%     ax1 = axes('Position',[0.7,0.5,0.2,0.2]); % weight = 0.3
    plot(x,y1,'Color',0.7*ones(1,3),'linewidth',2); hold on; box off; 
    id=abs(x-vp)==min(abs(x-vp)); xx1=x(id); yy1 = y1(id); plot([xx1,xx1],[0,yy1],'Color',0.7*ones(1,3)); 
%     text(8,0.15,['\lambda=',num2str(yc1,'%.2f')],'FontSize',20); 
%     text(8,0.2,'\lambda=\itf_{VP}\rm(\vartheta_{Vis})','FontSize',14); 
    text(8,0.2,'\lambda_{VP}'); 
    p1 = poisspdf(round(fr3(idx1)),yc1); x1=round(fr3(idx1)); plot([x1,x1],[0,p1],'k'); 
    plot([0,x1],[p1,p1],'k'); text(2/x1,0.3,'\bf{pr1}'); plot(x1,0,'.','Color',clrs(8,:),'markersize',10); 

    ax2 = axes('Position',[0.2,0.7,0.2,0.2]);
    plot(x,y2,'Color',0.3*ones(1,3),'linewidth',2); hold on; box off; 
    id=abs(x-hp)==min(abs(x-hp)); xx2=x(id); yy2 = y1(id); plot([xx2,xx2],[0,yy2],'Color',0.7*ones(1,3)); 
%     text(8,0.15,'\lambda=\itf_{P}\rm(\vartheta_{Pro})','FontSize',14); 
    text(8,0.15,'\lambda_{P}'); 
    p2 = poisspdf(x1,yc2); plot([x1,x1],[0,p2],'k'); 
    plot([0,x1],[p2,p2],'k'); text(2/x1,0.25,'\bf{pr2}'); plot(x1,0,'.','Color',clrs(8,:),'markersize',10); 
       
    id = (abs(cond3(:,1))>30 & abs(cond3(:,3))<20); 
    grp1 = cond3(~id,:); grp2 = cond3(id,:); grp2 = sortrows(grp2,6); grp2 = flipud(grp2); 
    figure; colormap(flipud(brewermap([],'RdBu'))); 
    plot([-55,55],[0,0],'k','linewidth',2); hold on; plot(-55:55,-55:55,'Color',0.7*ones(1,3),'linewidth',2); jit=1.2;
    scatter(grp1(:,1)+jit*randn(size(grp1,1),1),grp1(:,3),40,grp1(:,6),'filled','MarkerEdgeColor','w'); hold on;
    for ic = 1:length(grp2)
        scatter(grp2(ic,1)+jit*randn(1),grp2(ic,3),40,grp2(ic,6),'filled','MarkerEdgeColor','w'); 
    end
%     alpha(0.5); 
    xlabel('Disparity (deg)'); ylabel('Drift (deg)'); c=colorbar; c.Label.String = 'VP weight';   
    box off; xlim([-55,55]); ylim([-55,55]); % view([21,3]); grid off; 
    set(gca,'FontName','Calibri','FontSize',14,'FontWeight','bold'); 
    set(gca,'XTick',-40:20:40,'YTick',-40:20:40); 
%     set(gcf,'Position',[1192         452         344         252]); 
    set(gcf,'Position',[1192         415         380         289]); 
%     title('Data from large conflict'); xlabel('Drift'); ylabel('VP weight'); 
    
end
%% Fig 3F
modelFun = @(p,x)(((1 + cos(x/180*pi))/2).^p(1) + p(2));
figure();
hold on;
for i = 1:9
    x1 = grp1(grp1(:,1)==rots(i),6);
    x2 = grp2(grp2(:,1)==rots(i),6);
    xmean(i) = nanmean([x1;x2]);
end
scatter(rots, xmean)
nlModel = fitnlm(rots,xmean,modelFun,[0,0.2]);
xgrid = linspace(-80,80,100)';
line(xgrid,predict(nlModel,xgrid),'Color','r');
%% Fig 3E
figure; 
plot([-55,55],[0,0],'k','linewidth',2); hold on; 
plot(-55:55,-55:55,'Color',0.7*ones(1,3),'linewidth',2);
scatter(grp1(:,1),grp1(:,3),20,'k','filled'); hold on;
scatter(grp2(:,1),grp2(:,3),20,'k','filled'); 
scatter(cond3(12,1), cond3(12,3),10,'r','filled'); 

cond3 = cond3(:,[1,3]);
rot = unique(cond3(:,1));
yy = nan(10000,length(rot));
for i = 1:length(rot)
    x = cond3(cond3(:,1)==rot(i),2);
    yy(1:length(x),i) = x;
end
yy = nanmean(yy,1)';
% plot(rot,yy);
p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',4)
set(gca,'Position',[.1 .1 .4 .4]);
