%% Figure 2C sub6
load('hw_normalized_c3') 
rot = [-35,-30,-20,-10,0,10,20,30,35]';
a = [6]; 
figure()
hold on
yy = nan(length(rot),length(a));
k=1;
for sub = a
    dat = hw(hw(:,1)==sub,:);   % choose one out of 8 subjects
    cond3 = dat(dat(:,2)==1,[5,13]); rots = unique(cond3(:,1)); drf = nan(length(rots),100);
    for i = 1:length(rot)
        x = cond3(cond3(:,1)==rot(i),2);
        x = x(~isnan(x));
        x = -x;

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
        delind = delOutliers(x);
        x(delind) = nan;
        scatter(repmat(rot(i),length(x),1),x,10,'k','filled');
        yy(i,k) = nanmean(x);
        
    end  
    k = k+1;
end
plot([-40,40],[-40,40],'LineWidth',4)
plot([-35,35],[0,0],'LineWidth',4)  

p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',3)
ylim([-35,35])
set(gca,'Position',[.1 .1 .4 .4]);
%% Figure 2C sub6, simulation
load('HW_sym_meanpc_hand_p15v10_2.mat')
simudata = Data{1}{6}{2};
rot = [-35, -30,-20,-10,0,10,20,30, 35]';
figure(2);
hold on
idx = randsample([1:5000],15);
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
%% Figure 2C sub 1
load('hw_normalized_c3') 
rot = [-35,-30,-20,-10,0,10,20,30,35]';
a = [1]; 
figure()
hold on
yy = nan(length(rot),length(a));
k=1;
for sub = a
    dat = hw(hw(:,1)==sub,:);   % choose one out of 8 subjects
    cond3 = dat(dat(:,2)==1,[5,13]); rots = unique(cond3(:,1)); drf = nan(length(rots),100);
    for i = 1:length(rot)
        x = cond3(cond3(:,1)==rot(i),2);
        x = x(~isnan(x));
        x = -x;

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
        delind = delOutliers(x);
        x(delind) = nan;
        scatter(repmat(rot(i),length(x),1),x,10,'k','filled');
        yy(i,k) = nanmean(x);
        
    end  
    k = k+1;
end
plot([-40,40],[-40,40],'LineWidth',4)
plot([-35,35],[0,0],'LineWidth',4)  

p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',3)
ylim([-35,35])
set(gca,'Position',[.1 .1 .4 .4]);
%% Figure 2C sub6, simulation
load('HW_sym_meanpc_hand_p15v10_2.mat')
simudata = Data{1}{1}{2};
rot = [-35, -30,-20,-10,0,10,20,30, 35]';
figure(2);
hold on
idx = randsample([1:5000],15);
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
