setwd("D:\Projects\MI\\BCI_data\\Fig1");
library(R.matlab); library(ggplot2);library(plyr);library(multcomp);library(phia);
library(reshape2); library(lattice); library(nlme);library(lme4);#library (VennDiagram)
readMat("qn.mat")->qn
dat <- qn[[1]]
# dat <- matrix(dat)
dat <- data.frame(dat)
names(dat) <- c('sub','hw','rot','o','oc','a','ac')
dat_sel <- subset(dat, hw==1, select=sub:oc)
rat_mean <- aggregate(dat_sel[, 1:5], list(dat_sel$rot), mean)
rat_ste <- aggregate(dat_sel[, 1:5], list(dat_sel$rot), sd)/sqrt(17)

dat_sel_w <- subset(dat, hw==2, select=sub:oc)
rat_mean_w <- aggregate(dat_sel_w[, 1:5], list(dat_sel_w$rot), mean)
rat_ste_w <- aggregate(dat_sel_w[, 1:5], list(dat_sel_w$rot), sd)/sqrt(17)

rat_mean$ow = rat_mean_w$o
rat_ste$ow = rat_ste_w$o
#postscript('Rplot06.pdf',width=6,height=4)
ggplot(rat_mean, aes(x = Group.1, y= o), xlab="Disparity") +
  geom_bar(aes(x = Group.1, y= o,  fill='Ownership', col='#3A5FCD'), stat="identity", width=2, position = "dodge")+
  geom_errorbar(aes(ymin=rat_mean$o-rat_ste$o, ymax=rat_mean$o+rat_ste$o),width=1,position = "dodge",col = '#3A5FCD')+
  
  geom_bar(aes(x = Group.1+2, y= ow, fill='Control', col='gray'), stat="identity", width=2, position = "dodge")+
  geom_errorbar(aes(x = Group.1+2, ymin=rat_mean$ow-rat_ste$ow, ymax=rat_mean$ow+rat_ste$ow),width=1,position = "dodge", col = 'gray')+
  scale_fill_manual("", values = c('Ownership'="#3A5FCD", 'Control'="#41c1f0")) +
  scale_color_manual("", values = c('Ownership'="#3A5FCD", 'Control'="#41c1f0"))+
  
  xlab('Disparity') + ylab('Score') + coord_cartesian(xlim = c(-40, 40), ylim = c(-3,3))+ 
  scale_x_continuous(breaks=seq(-40, 40, 10)) + scale_y_continuous(breaks=seq(-3, 3, 1))+
  theme_bw()+theme(panel.grid=element_blank(),panel.border=element_blank())+
  theme(axis.text.x=element_text(size=14,face="bold"),axis.text.y=element_text(size=14,face="bold"),
        axis.title.x=element_text(size=18,face="bold"),axis.title.y=element_text(size=18,face="bold"))
#dev.off()

