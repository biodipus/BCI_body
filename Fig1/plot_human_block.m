%% Figure 1E
load('human_block_raw.mat') 
rot = [-35,-30,-20,-10,0,10,20,30,35]';
a = [1,2,3,4,6,7,9,12:15,16,17:19,20,22];
figure()
hold on
yy = nan(length(rot),length(a));
k=1;
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
        
        scatter(repmat(rot(i),length(x),1),x,10,'k','filled');
        yy(i,k) = nanmean(x);
        
    end  
    k = k+1;
end
plot([-40,40],[-40,40],'LineWidth',4)
plot([-35,35],[0,0],'LineWidth',4)  

drf_all = [];
k = 1;
for sub = a
    scatter(rot,yy(:,k),10,[0.5,0.5,0.5],'filled');
    p = polyfit(rot,yy(:,k),3);
    x1 = linspace(rot(1),rot(end));
    y1 = polyval(p,x1);
    drf_all = [drf_all,y1'];
    plot(x1,y1,'LineWidth',2,'color','b')
    k = k+1;
end
y1 = nanmean(drf_all,2);
plot(x1,y1,'LineWidth',2,'color','r')
set(gca,'Position',[.1 .1 .5 .5]);

