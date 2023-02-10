%%  Fig 3C
files = dir('D:\Projects\MI\BCI_data20230118\Monkey_data\Neuorns\H\neu*.mat'); 
phase = {'Ref','Img','Init','Mov','Fin'}; 
% fig = figure('Position',[44, 726, 1753, 234]); 
for f = 64 % Fig 3B:162 3C:64; Fig S5:195;
%     subplot(10,10,f)
    figure(2)
    file = files(f).name; 
    load(['D:\Projects\MI\BCI_data20230118\Monkey_data\Neuorns\H\',file]); 
    dat = data.all; info = data.info; spks = data.spks;
    if isempty(dat)
        continue;
    end    
%     type = data.type; % theta = unique(type(:,[2,8]),'rows');
    info1 = info; info1(:,6) = ceil(info1(:,6)/9); 
    types = info1(:,[3,4,6,7]); 
    if sum(types(:,1)==1 & types(:,2)==6) == 0 || sum(types(:,1)==2) == 0 
        continue;
    end
    rots = unique(types(:,4)); tgts = unique(types(:,3)); % 1-9T
    
%     FR = []; drf = []; pos = []; 
%     for itrl = 1:size(dat,1)
%         trial = dat{itrl}; time = trial(:,2); time2 = trial(:,3); 
%         FR(itrl,1) = mean(trial(time<-0.5,10)); 
%         FR(itrl,2) = mean(trial(time>-0.5 & time<0,10));  
%         FR(itrl,3) = mean(trial(time>0 & time<0.5,10)); 
%         FR(itrl,4) = mean(trial(time>0.5 & time<1,10)); 
%         FR(itrl,5) = mean(trial(time2>-0.5 & time2<0,10));  
%         idx2 = find(abs(time2)==min(abs(time2)),1); 
% %         drf(itrl,1) = trial(idx2,7); pos(itrl,:) = trial(idx2,8:9); 
%     end
% %     drf = -drf;     
        
    conds = unique(types(:,[1,4]),'rows'); 
%     [~,loc] = ismember(types(:,[1,4]),conds,'rows'); 
    clrs = colormap(jet); 
    clrs=clrs(round(linspace(1,256,length(rots))),:); 
    clrs=[0.7*ones(1,3);0.3*ones(1,3);clrs];

%%% FR
lty = ['-.','-.',strsplit(repmat('- ',1,size(conds,1)-2))]; lw = [2,2,repmat(1.2,1,size(conds,1)-2)];
tgts = unique(types(types(:,1)==3,3)); 
% tgts = [3,5,7];
idx = ismember(types(:,3),tgts); dat = dat(idx,:); types=types(idx,:); spks=spks(idx); 
[~,id] = sortrows(types,[1,4]); dat=dat(id,:); types=types(id,:); spks=spks(id); 
[~,loc] = ismember(types(:,[1,4]),conds,'rows'); 
mfr = nan(length(conds),length(tgts)); sfr = mfr; 
% t = linspace(-1,1,160); 
t = linspace(-0.75,0.5,101); 
for icond = 1:size(conds,1)
%     if ~ismember(icond,[1:2,4:10])
%         continue;
%     end
%     idx = (loc==icond & ismember(types(:,3),[5,7,8])); 
    idx = (loc==icond & ismember(types(:,3),tgts)); 
    if sum(idx)==0
        continue;
    end
    d = dat(idx); fr = []; 
    spk_tmp = spks(idx);
    for itrl = 1:size(d,1)
        if length(spk_tmp{itrl})>=5
        trial = d{itrl}; % fr = [fr trial(1:160,end)]; 
        time2 = trial(:,2); t2 = find(abs(time2)==min(abs(time2)));
%         time2 = trial(:,3); t2 = find(abs(time2)==min(abs(time2))); 
        fr = [fr trial(t2-60:t2+40,end)];
%         fr = [fr trial(t2-80:t2,end)];
        end
    end
    mfr = mean(fr'); 
    plot(t,mfr,'Color',clrs(icond,:),'linestyle',lty{icond},'linewidth',lw(icond)); hold on; 
end
yrng = get(gca,'YLim'); 
fill([-0.5,-0.5,0,0],[0,yrng(2),yrng(2),0],0.8*ones(1,3),'edgecolor','w'); hold on; alpha(0.3); 
% box off; xlabel('Time (s)'); ylabel('FR (spks/s)'); % ylim([0,6]); 
set(gca,'XTick',[-1,-0.5,0]);%-1.5:0.5:0 
xlim([-0.75,0.7]); set(gca,'XTick',-0.5:0.5:0.5); 
set(gca,'FontName','Calibri','FontWeight','bold','FontSize',14);
set(gcf,'Position',[702   669   398   299]);      
% legend(['VP','P',strsplit(num2str(rots'))],'location','northwest','box','off'); 
end
%% Fig 3C
files = dir('D:\Projects\MI\BCI_data20230118\Monkey_data\Neuorns\H\neu*.mat'); 
phase = {'Ref','Img','Init','Mov','Fin'}; 
% fig = figure('Position',[44, 726, 1753, 234]); 
fig = figure; 
for f = 64
    file = files(f).name; 
    load(['D:\Projects\MI\BCI_data20230118\Monkey_data\Neuorns\H\',file]); 
    dat = data.all; info = data.info; spks = data.spks;
    if isempty(dat)
        continue;
    end    
    type = data.type; 
    info1 = info; info1(:,6) = ceil(info1(:,6)/9); 
    types = info1(:,[3,4,6,7]); types(:,5) = type(:,8);
    if sum(types(:,1)==1 & types(:,2)==6) == 0 || sum(types(:,1)==2) == 0 
        continue;
    end
    rots = unique(types(:,4));
    
    conds = unique(types(:,[1,4]),'rows'); 
    conds(:,3) = (conds(:,1)==2); 
    conds(:,4) = abs(conds(:,2)-1); conds2 = sortrows(conds,3:4); conds2 = flipud(conds2); 
    [~,idc] = ismember(conds2(:,1:2),conds(:,1:2),'rows'); 
%     [~,loc] = ismember(types(:,[1,4]),conds,'rows'); 
    clrs = colormap(jet); clrs=clrs(round(linspace(1,256,length(rots))),:); clrs=[0.7*ones(1,3);0.3*ones(1,3);clrs];
    clrs2 = clrs(idc,:); 
    %%% Rasters
    tgts = unique(types(types(:,1)==3,3)); 
%     tgts = [5,7,8];
    idx = ismember(types(:,3),tgts); dat = dat(idx,:); types=types(idx,:); spks=spks(idx); 
    [~,id] = sortrows(types,[1,4]); dat=dat(id,:); types=types(id,:); spks=spks(id); 
    
    [~,loc] = ismember(types(:,[1,4]),conds2(:,1:2),'rows'); 
    kk=1; 

    for icond = 1:size(conds,1)            
        idx = find(loc==icond); spk = spks(idx); 
%         spk = spk(randperm(length(spk),30));
        for trl = 1:length(spk)
            if length(spk{trl})>=5
            spk1 = spk{trl}; spk1=spk1(spk1<=1); x = repmat(spk1,1,2); % x=x+itg*2; 
            y = [zeros(size(x,1),1)+kk-1,zeros(size(x,1),1)+kk+1]; kk=kk+1;
            line(x',y','Color',clrs2(icond,:),'linewidth',1.5); hold on; % 
            end
        end  
    end
%     plot([-1,1],[kk+0.5,kk+0.5],'k'); kk=kk+1; 

    xlim([-1,0.5]); ylim([-1 kk+1]); 
    yrng = get(gca,'YLim'); 
    plot([0,0],[0,yrng(2)],'k'); 
    plot([-0.5,-0.5],[0,yrng(2)],'k'); 
    
    ylabel('Trial #'); xlabel('Time (s)'); 
    yrng=get(gca,'YLim'); plot([0,0],[0,yrng(2)],'k'); 
    set(gca,'FontName','Calibri','FontWeight','bold','FontSize',14,'XTick',-1:0.5:1); 
    set(gcf,'Position',[702   669   259   299]);
%     set(gcf,'Position',[1380 441 297 334]);
    
    print('-painters','-depsc','-r600','fig1'); 

end
