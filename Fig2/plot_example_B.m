%% Fig 2F
load('D:\Projects\MI\BCI_data\Fig1\B_beha_10example.mat')
rot = [-30,-20,-10,0,10,20,30]';
hw = reach;
days = unique(hw(:,1));
a = [7];
k = 1;
yy = cell(1,length(rot));
figure()
for sub = a
    hold on
    dat = hw(hw(:,1)==sub,:);   
    dat = dat(dat(:,3)==6,:); % hand 
    cond3 = dat(dat(:,2)==3,[5, 6]); 
    base = nanmean(cond3(cond3(:,1)==0,2));
    cond3(:,2) = cond3(:,2)-base;
    for i = 1:length(rot)
        x = cond3(cond3(:,1)==rot(i),2);
        x = x(~isnan(x));
        r = rot(i);

        if r<0
            idx = (x<r-7) | (x>7);
            x(idx) = nan;
        end
        if r>0
            idx = (x>r+7) | (x<-7);
            x(idx) = nan;
        end
        if r==0
            idx = (x>7) | (x<-7);
            x(idx) = nan;
        end
        scatter(repmat(rot(i),length(x),1),x,10,'k','filled');
        yy{i} = [yy{i};x];
    end

plot([-30,30],[-30,30],'LineWidth',3)
plot([-30,30],[0,0],'LineWidth',3)
k = k+1;
end

yy = cellfun(@nanmean,yy);
p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',3)
set(gca,'Position',[.1 .1 .4 .4]);
%% Fig 2G
load('meanpc_example_B.mat')
rot = [-30,-20,-10,0,10,20,30]';
figure(2);
hold on
idx = randsample([1:5000],30);
simu_sel = simudata(idx,:);
for i = 1:length(rot)
    x = simu_sel(:,i);
    r = rot(i);
    if r<0
        idx = (x<r-7) | (x>7);
        x(idx) = nan;
    end
    if r>0
        idx = (x>r+7) | (x<-7);
        x(idx) = nan;
    end
    if r==0
        idx = (x>7) | (x<-7);
        x(idx) = nan;
    end
    scatter(repmat(rot(i),length(x),1),x,15,'k','filled');
end
yy = nanmean(simu_sel)';
% plot(rot,yy);
p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',4)
plot([-45,45],[-45,45],'LineWidth',4)
plot([-45,45],[0,0],'LineWidth',4)
%% Fig 2H
load('meanpc_example_B.mat')
figure(3);
hold on
plot(rot, pc1mean,'color','r','marker','o','linewidth',5)
set(gca,'XTick',[-20,0,20]);
set(gca,'FontName','Calibri','FontWeight','bold','FontSize',14);
xlabel('Disparity (deg)'); ylabel('Pcom');




