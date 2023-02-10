% fig 4C
load('D:\Projects\MI\BCI_data\Fig1\H_beha_10example.mat')
rot = [-45,-35,-20,-10,0,10,20,35,45]';
hw = reach;
days = unique(hw(:,1));
a = [3];
k = 1;
yy = cell(1,length(rot));
figure();
hold on
for sub = a
    dat = hw(hw(:,1)==sub,:);
    dat = dat(dat(:,4)==1,:);
    dat = dat(dat(:,2)==6,:);
    cond3 = dat(dat(:,3)==3,[6, end]);
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
        delind = delOutliers(x);
        x(delind) = nan;
        scatter(repmat(rot(i),length(x),1),x,10,'k','filled');
        yy{i} = [yy{i};x];
    end
    k = k+1;
end

yy = cellfun(@nanmean,yy);
p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',3)
set(gca,'Position',[.1 .1 .6 .7]);

%%%%%%%% wood
k = 1;
yy = cell(1,length(rot));

for sub = a
    dat = hw(hw(:,1)==sub,:);
    dat = dat(dat(:,4)==1,:);
    dat = dat(dat(:,2)==21,:);
    cond3 = dat(dat(:,3)==3,[6, end]);
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
        delind = delOutliers(x);
        x(delind) = nan;
        scatter(repmat(rot(i),length(x),1),x,10,'b','filled');
        yy{i} = [yy{i};x];
    end
    k = k+1;
end

yy = cellfun(@nanmean,yy);
p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',3)
set(gca,'Position',[.1 .1 .6 .7]);
ylim([-35,35])