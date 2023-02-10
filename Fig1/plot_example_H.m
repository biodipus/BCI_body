%% fig S1B
load('H_beha_10example.mat')
rot = [-45,-35,-20,-10,0,10,20,35,45]';
hw = reach;
days = unique(hw(:,1));
a = [1:10]; 
k = 1;
yy = zeros(length(rot),length(a));
figure()
for sub = a
    yy = cell(1,length(rot));
    subplot(2,5,k)
    hold on 
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

        scatter(repmat(rot(i),length(x),1),x,10,'k','filled');
        yy{i} = [yy{i};x];
    end


    plot([-45,45],[-45,45],'LineWidth',3)
    plot([-45,45],[0,0],'LineWidth',3)

    yy = cellfun(@nanmean,yy);
    p = polyfit(rot,yy,3);
    x1 = linspace(rot(1),rot(end));
    y1 = polyval(p,x1);
    plot(x1,y1,'LineWidth',4)

    k = k+1;
end
set(gcf,'unit','normalized','position',[0.2,0.2,0.64,0.32]);
%% fig 1F
load('H_beha_10example.mat')
rot = [-45,-35,-20,-10,0,10,20,35,45]';
hw = reach;
days = unique(hw(:,1));
a = [2:5];
k = 1;
yy = cell(1,length(rot));
figure();
hold on
for sub = a
    dat = hw(hw(:,1)==sub,:);
    dat = dat(dat(:,4)==1,:);
    dat = dat(dat(:,2)==6,:); 
    cond3 = dat(dat(:,3)==3,[6, end]);
    base = nanmean(cond3(cond3(:,1)==0,2));
    cond3(:,2) = cond3(:,2)-base;
    for i = 1:length(rot)
        x = cond3(cond3(:,1)==rot(i),2);
        x = x(~isnan(x));
        r = rot(i);
        if r==0
            nanmean(x)
        end
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
    k = k+1;
end

plot([-45,45],[-45,45],'LineWidth',4)
plot([-45,45],[0,0],'LineWidth',4)

yy = cellfun(@nanmean,yy);
p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',3)
set(gca,'Position',[.1 .1 .4 .4]);
%% fig S1E
load('D:\Projects\MI\BCI_data\Monkey_data\Behavior\H_beha_sessions.mat')
rot = [-90,-45,-35,-20,-10,0,10,20,35,45,90]';
hw = reach;
days = unique(hw(:,1));
a = [72:74]; 
k = 1;
yy = cell(1,length(rot));
figure();
hold on
for sub = a
    dat = hw(hw(:,1)==sub,:);   
    cond3 = dat(dat(:,2)==3,[5, 6]); 
    for i = 1:length(rot)
        x = cond3(cond3(:,1)==rot(i),2);
        x = x(~isnan(x));
        r = rot(i);
        scatter(repmat(rot(i),length(x),1),x,10,'k','filled');
        yy{i} = [yy{i};x];
    end
    k = k+1;
end

plot([-90,90],[-90,90],'LineWidth',4)
plot([-90,90],[0,0],'LineWidth',4)

yy = cellfun(@nanmean,yy);
p = polyfit(rot,yy,3);
x1 = linspace(rot(1),rot(end));
y1 = polyval(p,x1);
plot(x1,y1,'LineWidth',3)
set(gca,'Position',[.1 .1 .4 .5]);

