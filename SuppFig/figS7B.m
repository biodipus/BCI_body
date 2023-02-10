%% Fig S7B
pvals = nan(5,2); rots = [-45,-35,-20:10:20,35,45]';
load('ctrlReg2H.mat'); lik1H = likeli; load('ctrlReg2N.mat'); lik1N = likeli;
xx = (-60:10:60)'; edges=(xx(2:end)+xx(1:end-1))/2; xx = (edges(2:end)+edges(1:end-1))/2; 
xind = (1:length(xx))'; [xgrid,ygrid] = meshgrid(rots,xind); conds = [xgrid(:) ygrid(:)];
% edges = [-45,-35,-25,-15,-6,6,15,25,35,45]'; xx = edges; xx(1)=-45; xx(end) = 45; xx=(xx(2:end)+xx(1:end-1))/2; 
% xind = (1:length(xx))'; [xgrid,ygrid] = meshgrid(rots,xind); conds = [xgrid(:) ygrid(:)];

clrs = colormap(jet); clrs=clrs(round(linspace(1,64,length(rots))),:); 
labs = {'pHand position (deg)','vHand position (deg)','Eye position (deg)','pHand-Eye distance (deg)','vHand-Eye distance (deg)'};
for isub = 1:5
    subplot(2,3,isub); 
    for ic = 1:length(xx)
        scatter(xx(ic)*ones(1,9),lik1H(isub,conds(:,2)==ic),30,clrs,'o','filled'); hold on; alpha(0.5); 
        scatter(xx(ic)*ones(1,9),lik1N(isub,conds(:,2)==ic),30,clrs,'^','filled'); hold on; alpha(0.5); 
    end
    mdl1 = fitlm(xx(conds(:,2)),lik1H(isub,:)','linear','RobustOpt','on'); yfit1 = feval(mdl1,xx); 
    h1 = plot(xx,yfit1,'Color',0.2*ones(1,3),'linewidth',2); hold on;
    mdl2 = fitlm(xx(conds(:,2)),lik1N(isub,:)','linear','RobustOpt','on'); yfit2 = feval(mdl2,xx); 
    h2 = plot(xx,yfit2,'Color',0.2*ones(1,3),'linewidth',2,'linestyle','--'); hold on;
    set(gca,'FontName','Calibri','FontSize',18,'FontWeight','bold'); xlabel(labs{isub}); ylim([0,1]);
    if isub == 1
        legend([h1,h2],{'H','N'},'box','off'); 
    end
    if isub == 1
        ylabel('VP weight'); 
    else 
        ylabel('');
    end
    pvals(isub,1) = table2array(mdl1.Coefficients(2,4)); pvals(isub,2) = table2array(mdl2.Coefficients(2,4)); 
end
set(gcf,'Position',[128         351        1112         627]); 
%%