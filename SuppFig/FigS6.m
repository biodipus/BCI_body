load cmpPC; 
rots = cmpPC.rots; pc1 = cmpPC.pcH; pc2 = cmpPC.pcN; ml1 = cmpPC.weightH; ml2 = cmpPC.weightN; 
pcfit1 = cmpPC.pcfit1; pcfit2 = cmpPC.pcfit2; wfit1 = cmpPC.wfit1; wfit2 = cmpPC.wfit2; 
%% Fig S6A
fig = figure; left_color = [178 34 34]/255; right_color = [58 95 205]/255;
set(fig,'defaultAxesColorOrder',[left_color; right_color]);
yyaxis left; 
h1=plot(rots,pc1,'^','markeredgecolor',[178 34 34]/255,'markerfacecolor',[178 34 34]/255); hold on; 
plot(-45:45,pcfit1,'Color',[178 34 34]/255,'linewidth',1.5,'linestyle','-'); 
h2=plot(rots,pc2,'o','markeredgecolor',[178 34 34]/255,'markerfacecolor',[178 34 34]/255);
plot(-45:45,pcfit2,'Color',[178 34 34]/255,'linewidth',1.5,'linestyle','-.'); box off; 
set(gca,'FontName','Calibri','FontWeight','bold','FontSize',18); xlim([-55,55]);
xlabel('Disparity (deg)'); ylabel('Pcom'); set(gcf,'Position',[1192  415   344    289]); 
% legend([h1,h2],{'H','N'},'box','off'); 
ylim([0.4,1.2]);

yyaxis right; 
h21=plot(rots,ml1,'o','markeredgecolor',[58 95 205]/255,'markerfacecolor',[58 95 205]/255); hold on; 
plot(-45:45,wfit1,'Color',[58 95 205]/255,'linewidth',1.5,'linestyle','-'); 
h22=plot(rots,ml2,'^','markeredgecolor',[58 95 205]/255,'markerfacecolor',[58 95 205]/255); hold on; 
plot(-45:45,wfit2,'Color',[58 95 205]/255,'linewidth',1.5,'linestyle','-.'); box off; 
set(gca,'FontName','Calibri','FontWeight','bold','FontSize',18); xlim([-55,55]);
xlabel('Disparity (deg)'); ylabel('VP weight'); set(gcf,'Position',[1192  415   344    289]); 
legend([h21,h22],{'H','N'},'box','off'); 
ylim([0.48,0.56]);

%% Fig S6B
clrs = colormap(jet); clrs = clrs(round(linspace(1,64,length(rots))),:); 
figure('Position',[1192  415   344    289]); 
xx = 0.4:0.01:0.7; 
[mdl1,gof1] = fit(ml1,pc1,'poly1'); yfit1 = feval(mdl1,xx); 
mdl1 = fitlm(ml1,pc1); 
h1 = plot(xx,yfit1,'Color',0.7*ones(1,3),'linewidth',2); hold on;
[mdl2,gof2] = fit(ml2,pc2,'poly1'); yfit2 = feval(mdl2,xx); 
mdl2 = fitlm(ml2,pc2); 
h2 = plot(xx,yfit2,'Color',0.7*ones(1,3),'linewidth',2,'linestyle','--'); 

scatter(ml1,pc1,30,clrs,'o','filled'); hold on; 
scatter(ml2,pc2,30,clrs,'^','filled'); hold on; box off; 

set(gca,'FontName','Calibri','FontWeight','bold','FontSize',14);
xlabel('VP weight'); ylabel('P(C=1|Data)'); set(gcf,'Position',[1192  415   344    289]); 
legend([h1,h2],{'H','N'},'box','off','location','northwest'); % xlim([0.45,0.6]); ylim([0,1.2]);
xlim([0.45,0.58]); ylim([0,1.3]);
