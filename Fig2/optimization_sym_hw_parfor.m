clear all;clc
fitname = {'average';'select';'match';};
reach=[];
if isempty(reach)
    load .\Monkey_data\S_beha_days;
end

for c = 1
    subnum = unique(reach(:,1))';
    for sub = 1:15

        dat = reach(reach(:,1)==sub,:);   

        cond3 = dat([abs(dat(:,5))<90 & dat(:,2)==3 & dat(:,3)==6],[5,6]);
        cond3w = dat([abs(dat(:,5))<90 & dat(:,2)==3 & dat(:,3)==21],[5,6]);
        
        rots = unique(cond3(:,1));
        drf = nan(length(rots),300); %drfsd=drf;
        if ~isempty(cond3)
            for i = 1:length(rots)
                %plot(cond3(cond3(:,1)==rots(i),1),cond3(cond3(:,1)==rots(i),2),'b.'); hold on;
                aa = cond3(cond3(:,1)==rots(i),2);
                aa = aa(~isnan(aa));
                delind = delOutliers(aa);
                aa(delind) = nan;
                drf(i,1:length(aa)) = aa;

            end
            realdata_h = drf';
            
            if ~isempty(cond3w)
                rots = unique(cond3w(:,1));
                drf = nan(length(rots),300); %drfsd=drf;
                for i = 1:length(rots)
                    %plot(cond3(cond3(:,1)==rots(i),1),cond3(cond3(:,1)==rots(i),2),'b.'); hold on;
                    aa = cond3w(cond3w(:,1)==rots(i),2);
                    aa = aa(~isnan(aa));
                    delind = delOutliers(aa);
                    aa(delind) = nan;
                    drf(i,1:length(aa)) = aa;
                    %     drfsd(i,1) = nanstd(cond3(cond3(:,1)==rots(i),2));
                end
                realdata_w = drf';
            else
                realdata_w = [];
            end
            
        end
        %         obenum(sub,1) = length(find(~isnan(realdata(:,rots<=0))));
        if ~isempty(cond3w)
            fun = @(params)(fitness_sym_HW_parfor(params,realdata_h, realdata_w,rots,c));
            [r1,r2,r3,r4,r5,r6] = ga(fun,4,[-1,1,0,0],0,[],[],[1,1,0,0],[15,10,10,10]);
        else
            fun = @(params)(fitness_sym_H_parfor(params,realdata_h,rots,c));
            [r1,r2,r3,r4,r5,r6] = ga(fun,3,[-1,1,0],0,[],[],[1,1,0],[15,10,10]);
        end
        obenum(sub,1) = length(find(~isnan(realdata_h))) + length(find(~isnan(realdata_w)));
        finalp{sub,1} = r1;
        fval{sub,1} = r2;
        exitflag{sub,1} = r3;
        output{sub,1} = r4;
        population{sub,1} = r5;
        scores{sub,1} = r6;
%         [sub,finalp{sub,1}]
    end
    save(['.\causal_fit\HW_Norc3_sym_15day_',fitname{c,1},'_p15v10_single',num2str(subnum(end))],'finalp','fval','exitflag','output','population','scores','obenum');
    %     clearvars -except fitname c reach selectneuron
end