%% fig S1A
load('B_beha_10example.mat')
rot = [-30,-20,-10,0,10,20,30]';
hw = reach;
days = unique(hw(:,1));
a = days(1:10)'; 
k = 1;
figure()
for sub = a
    yy = cell(1,length(rot));
    subplot(2,5,k)
    hold on
    dat = hw(hw(:,1)==sub,:);   
    dat = dat(dat(:,3)==6,:); % hand 
    cond3 = dat(dat(:,2)==3,[5, 6]); 
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

yy = cellfun(@nanmean,yy);
p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',4)

plot([-30,30],[-30,30],'LineWidth',3)
plot([-30,30],[0,0],'LineWidth',3)
k = k+1;
end
%% Fig 1F
load('B_beha_10example.mat')
rot = [-30,-20,-10,0,10,20,30]';
hw = reach;
days = unique(hw(:,1));
a = [7:10]; 
k = 1;
yy = cell(1,length(rot));
figure()
for sub = a
    hold on
    dat = hw(hw(:,1)==sub,:);   
    dat = dat(dat(:,3)==6,:); % hand 
    cond3 = dat(dat(:,2)==3,[5, 6]); 
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
set(gca,'Position',[.1 .1 .4 .5]);


