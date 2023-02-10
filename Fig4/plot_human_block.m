%% Fig 4C human hand wood
load('hw') 
rot = [-35,-30,-20,-10,0,10,20,30,35]';
a = [1:17];
figure()
hold on
drf_h = nan(length(rot),length(a));
drf_w = nan(length(rot),length(a));
k = 1;
for sub = a
    dat = hw(hw(:,1)==sub,:);   
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
        
        drf_h(i,k) = nanmean(x);
        
    end 
    k=k+1;
end
k = 1;
for sub = a
    dat = hw(hw(:,1)==sub,:);   
    cond3 = dat(dat(:,2)==2,[5,13]); rots = unique(cond3(:,1)); drf = nan(length(rots),100);
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
        
        drf_w(i,k) = nanmean(x);
        
    end 
    k=k+1;
end

drf_all = [];
k = 1;
for sub = a
    scatter(rot,drf_h(:,k),10,'k','filled');
    p = polyfit(rot,drf_h(:,k),3);
    x1 = linspace(rot(1),rot(end));
    y1 = polyval(p,x1);
    drf_all = [drf_all,y1'];
    k=k+1;
end
y1 = nanmean(drf_all,2);
plot(x1,y1,'LineWidth',2,'color','k')

drf_all = [];
k=1;
for sub = a
    scatter(rot,drf_w(:,k),10,'b','filled');
    p = polyfit(rot,drf_w(:,k),3);
    x1 = linspace(rot(1),rot(end));
    y1 = polyval(p,x1);
    drf_all = [drf_all,y1'];
    k=k+1;
end
y1 = nanmean(drf_all,2);
plot(x1,y1,'LineWidth',2,'color','b')
set(gca,'Position',[.1 .1 .5 .5]);
