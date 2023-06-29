
library("stats")
library("Rgraphviz")
#install.packages("remotes")
#BiocManager::install("Rgraphviz")
library("fitdistrplus")
library("bnlearn")
library("readxl")
library("hdrcde")
library(gRain)
library(stringi)
library("greekLetters")
library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(stringr)
# library(installr)
#greeks('beta')
#source("http://bioconductor.org/biocLite.R")
#biocLite(c("graph", "Rgraphviz"))
setwd("C:/Users/20204066/Box/RESS-format/2022Dec10-RunFullmodel/18December2022SSBN4RYesting/")
getwd()

BNLayerOneTime = model2network(paste0("[MR1][MR2][R12Avl|MR1][R13Avl|MR2][SysTT|R12Avl:R13Avl]",
                                      ""))
#graphviz.plot(BNLayerOneTime, highlight = 
#                list(nodes=nodes(BNLayerOneTime), fill="lightblue", col="black"))#layout = "dot")


CondiProbTableLevels21 <- read.csv('conditional_prob_table_road_level_concate01edit.csv',sep = ",",2)                                                                             
View(CondiProbTableLevels21)
DFCondiProbTableLevels21 = data.frame(CondiProbTableLevels21)
DFCondiProbTableLevels21$DepenVar


CondiProbTableLevels10 <- read.csv('conditional_prob_table_road_to_subnet_level_concate01edit.csv',sep = ",",2)                                                                             
View(CondiProbTableLevels10)
DFCondiProbTableLevels10 = data.frame(CondiProbTableLevels10)
DFCondiProbTableLevels10$DepenVar



CondiProbTableLevelsSystem <- read.csv('conditional_prob_table_system_level.csv',sep = ",",2)                                                                             
View(CondiProbTableLevelsSystem)
DFCondiProbTableLevelsSystem = data.frame(CondiProbTableLevelsSystem)
DFCondiProbTableLevelsSystem$DepenVar

RoadMaintenanceVar <-as.list(unique(CondiProbTableLevels21$RoadMaintenanceRate))
RoadVar <- as.list(unique(DFCondiProbTableLevels21$DepenVar))
RoadMaintenanceVar
model_string <- ""
for (j in RoadMaintenanceVar){
  assign(as.vector(paste0(j, 'Range')), c(0,1))
  #road_prob_string <- ""
  #road_prob_string <- paste(road_prob_string, paste0('MRProb', j), "=", " array(c(0.6,0.4), dim = 2, dimnames = list(", paste0(j),"=", paste0(j, 'Range'), "))")
  model_string <- paste0(model_string, "[", j, "]")
  #write(road_prob_string,file="Text.txt",append=TRUE)
  }


MRProbRoad13MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road13MaintenanceRate = Road13MaintenanceRateRange ))
MRProbRoad40MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road40MaintenanceRate = Road40MaintenanceRateRange ))
MRProbRoad22MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road22MaintenanceRate = Road22MaintenanceRateRange ))
MRProbRoad24MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road24MaintenanceRate = Road24MaintenanceRateRange ))
MRProbRoad37MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road37MaintenanceRate = Road37MaintenanceRateRange ))
MRProbRoad49MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road49MaintenanceRate = Road49MaintenanceRateRange ))
MRProbRoad50MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road50MaintenanceRate = Road50MaintenanceRateRange ))
MRProbRoad57MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road57MaintenanceRate = Road57MaintenanceRateRange ))
MRProbRoad43MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road43MaintenanceRate = Road43MaintenanceRateRange ))
MRProbRoad58MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road58MaintenanceRate = Road58MaintenanceRateRange ))
MRProbRoad63MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road63MaintenanceRate = Road63MaintenanceRateRange ))
MRProbRoad13MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road13MaintenanceRate = Road13MaintenanceRateRange ))
MRProbRoad40MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road40MaintenanceRate = Road40MaintenanceRateRange ))
MRProbRoad22MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road22MaintenanceRate = Road22MaintenanceRateRange ))
MRProbRoad24MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road24MaintenanceRate = Road24MaintenanceRateRange ))
MRProbRoad37MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road37MaintenanceRate = Road37MaintenanceRateRange ))
MRProbRoad49MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road49MaintenanceRate = Road49MaintenanceRateRange ))
MRProbRoad50MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road50MaintenanceRate = Road50MaintenanceRateRange ))
MRProbRoad57MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road57MaintenanceRate = Road57MaintenanceRateRange ))
MRProbRoad43MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road43MaintenanceRate = Road43MaintenanceRateRange ))
MRProbRoad58MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road58MaintenanceRate = Road58MaintenanceRateRange ))
MRProbRoad63MaintenanceRate =  array(c(0.6,0.4), dim = 2, dimnames = list( Road63MaintenanceRate = Road63MaintenanceRateRange ))




for (j in RoadMaintenanceVar){
  for (i in RoadVar){
    if (grepl(substr(j,0,6), i)){
      model_string <- paste0(model_string, "[", i, "|", j, "]")
      }
    }
  }
model_string
BNLayerOneTime = model2network(model_string)
BNLayerOneTime

for (i in RoadVar) {
  road_prob_string <- ""
  assign(as.vector(paste0(i)), c(0,1))
  assign(as.vector(paste0('MRRange', i)), c(0,1))
  subset <-subset(DFCondiProbTableLevels21, DepenVar==i)
  levels<-c(unique(subset$IndepenVarLvel))
  assign(as.vector(paste0(i, 'Range')), levels)
  road_prob_string <- paste(road_prob_string, paste0('MRProb', i), "=", " array(c(0.6,0.4), dim = 3, dimnames = list(", paste0(i),"=", paste0(i, 'Range'), "))")
  #assign(get(paste0('MRProb', i)), array(c(0.6,0.4), dim = 3, dimnames = list(eval(paste0(i))= as.vector(paste0(i, 'Range')))))
  #road_prob_string <- paste0(road_prob_string, "\n")
  #cat(paste(road_prob_string, sep="\n"))
  write(road_prob_string,file="Text.txt",append=TRUE)
}

MRProbRoad13Avail11<-DFCondiProbTableLevels21[DFCondiProbTableLevels21$DepenVar=="Road13Avail","CondProb"]
MRProbRoad13Avail<-array(MRProbRoad13Avail, dim = c(2, 2),
                  dimnames = list(Road13Avail= Road13AvailRange, Road13MaintenanceRate = Road13MaintenanceRateRange))

#the above is working, adapt to loop
MRProbRoad13Avail =  array(c(0.6,0.4), dim = 2, dimnames = list( Road13Avail = Road13AvailRange ))
MRProbRoad13Costs =  array(c(0.6,0.4), dim = 2, dimnames = list( Road13Costs = Road13CostsRange ))
MRProbRoad40Avail =  array(c(0.6,0.4), dim = 3, dimnames = list( Road40Avail = Road40AvailRange ))
MRProbRoad40Costs =  array(c(0.6,0.4), dim = 3, dimnames = list( Road40Costs = Road40CostsRange ))
MRProbRoad22Avail =  array(c(0.6,0.4), dim = 2, dimnames = list( Road22Avail = Road22AvailRange ))
MRProbRoad22Costs =  array(c(0.6,0.4), dim = 2, dimnames = list( Road22Costs = Road22CostsRange ))
MRProbRoad24Avail =  array(c(0.6,0.4), dim = 3, dimnames = list( Road24Avail = Road24AvailRange ))
MRProbRoad24Costs =  array(c(0.6,0.4), dim = 3, dimnames = list( Road24Costs = Road24CostsRange ))
MRProbRoad37Avail =  array(c(0.6,0.4), dim = 2, dimnames = list( Road37Avail = Road37AvailRange ))
MRProbRoad37Costs =  array(c(0.6,0.4), dim = 2, dimnames = list( Road37Costs = Road37CostsRange ))
MRProbRoad49Avail =  array(c(0.6,0.4), dim = 3, dimnames = list( Road49Avail = Road49AvailRange ))
MRProbRoad49Costs =  array(c(0.6,0.4), dim = 3, dimnames = list( Road49Costs = Road49CostsRange ))
MRProbRoad50Avail =  array(c(0.6,0.4), dim = 2, dimnames = list( Road50Avail = Road50AvailRange ))
MRProbRoad50Costs =  array(c(0.6,0.4), dim = 2, dimnames = list( Road50Costs = Road50CostsRange ))
MRProbRoad57Avail =  array(c(0.6,0.4), dim = 3, dimnames = list( Road57Avail = Road57AvailRange ))
MRProbRoad57Costs =  array(c(0.6,0.4), dim = 3, dimnames = list( Road57Costs = Road57CostsRange ))
MRProbRoad43Avail =  array(c(0.6,0.4), dim = 3, dimnames = list( Road43Avail = Road43AvailRange ))
MRProbRoad43Costs =  array(c(0.6,0.4), dim = 3, dimnames = list( Road43Costs = Road43CostsRange ))
MRProbRoad58Avail =  array(c(0.6,0.4), dim = 3, dimnames = list( Road58Avail = Road58AvailRange ))
MRProbRoad58Costs =  array(c(0.6,0.4), dim = 3, dimnames = list( Road58Costs = Road58CostsRange ))
MRProbRoad63Avail =  array(c(0.6,0.4), dim = 3, dimnames = list( Road63Avail = Road63AvailRange ))
MRProbRoad63Costs =  array(c(0.6,0.4), dim = 2, dimnames = list( Road63Costs = Road63CostsRange ))






MRProbRoad13Avail= array(c(0.6,0.4), dim = 2, dimnames = list(Road13Avail=Road13AvailRange))



road_prob_string

assign('MRProbRoad43Avail', array(c(0.6,0.4), dim = 3, dimnames = list(eval(paste0('Road43Avail'))= Road43AvailRange)))


list(Road43Avail= Road43AvailRange)
eval(paste0('Road43Avail'))
list(Road13Avail= eval(paste0(RoadVar[1], 'Range')))

(parse(RoadVar[1]))

eval(paste0('MRProb', RoadVar[1]))

assign(eval(paste0('MRProb', RoadVar[1])), array(c(0.6,0.4), dim = 3, dimnames = list(eval(paste0(RoadVar[1]))= eval(paste0(RoadVar[1], 'Range')) )))


list('MRRangeRoad43Avail'= Road43AvailRange)

assign(MRProbRoad13Avail, 
       array(c(0.6,0.4), dim = 3, dimnames = 
               list(paste0("'",eval(paste0('MRRange', RoadVar[1])), "'")= eval(paste0(RoadVar[1], 'Range')) )))


list( eval(paste0('MRVariable', RoadVar[1])) = eval(paste0(RoadVar[1], 'Range')) )

parse("'",eval(paste0('MRRange', RoadVar[1])),"'") 
     
     
MR1Range=c(0,1)
MR2Range=c(0,1)
R12AvlRange=c(0,1,2)
R13AvlRange=c(0,1,2)
sysTTRange=c(0,1,2)
MR1.prob = array(c(0.6,0.4), dim = 2, dimnames = list(MR1= MR1Range))
MR2.prob = array(c(0.6, 0.4), dim = 2, dimnames = list(MR2= MR2Range))


R12Avlprob11<-DFCondiProbTableLevels21[DFCondiProbTableLevels21$DepenVar=="Road1-2Avail","CondProb"]
R12Avlprob<-array(R12Avlprob11, dim = c(3, 2),
                    dimnames = list(R12Avl= R12AvlRange, MR1 = MR1Range))


R13Avlprob11<-DFCondiProbTableLevels21[DFCondiProbTableLevels21$DepenVar=="Road1-3Avail","CondProb"]
R13Avlprob<-array(R13Avlprob11, dim = c(3, 2),
                  dimnames = list(R13Avl= R13AvlRange, MR2 = MR2Range))



sysTTprob11<-DFCondiProbTableLevels10[DFCondiProbTableLevels10$DepenVar=="TravelTimeDisc11","CondProb"]
sysTTprob<-array(sysTTprob11, dim = c(3, 3, 3),
                       dimnames = list(SysTT= sysTTRange, R12Avl= R12AvlRange,R13Avl= R13AvlRange))


cpt=list(MR1=MR1.prob,MR2=MR2.prob,R12Avl=R12Avlprob,R13Avl=R13Avlprob,SysTT=sysTTprob)



bn11 = custom.fit(BNLayerOneTime, cpt)
#graphviz.chart(bn11,grid = TRUE)
graphviz.chart(bn11, type = "barprob",scale = c(1.25, 1.5), grid = TRUE, bar.col = "darkgreen",strip.bg = "white")

#Cost
  
BNLayerOneCost = model2network(paste0("[MR1][MR2][R12Cost|MR1][R13Cost|MR2][SysCost|R12Cost:R13Cost]",
                                      ""))
#graphviz.plot(BNLayerOneCost, highlight = 
#                list(nodes=nodes(BNLayerOneCost), fill="lightblue", col="black"))#layout = "dot")



R12CostRange=c(0,1,2)
R13CostRange=c(0,1,2)
sysCostRange=c(0,1,2)


R12Costprob11<-DFCondiProbTableLevels21[DFCondiProbTableLevels21$DepenVar=="Road1-2Costs","CondProb"]
R12Costprob<-array(R12Costprob11, dim = c(3, 2),
                  dimnames = list(R12Cost= R12CostRange, MR1 = MR1Range))

R13Costprob11<-DFCondiProbTableLevels21[DFCondiProbTableLevels21$DepenVar=="Road1-3Costs","CondProb"]
R13Costprob<-array(R13Costprob11, dim = c(3, 2),
                   dimnames = list(R13Cost= R13CostRange, MR2 = MR2Range))


sysCostprob11<-DFCondiProbTableLevels10[DFCondiProbTableLevels10$DepenVar=="TotalCostDisc11","CondProb"]
sysCostprob<-array(sysCostprob11, dim = c(3, 3, 3),
                 dimnames = list(SysCost= sysCostRange, R12Cost= R12CostRange,R13Cost= R13CostRange))

cptCost=list(MR1=MR1.prob,MR2=MR2.prob,R12Cost=R12Costprob,R13Cost=R13Costprob,SysCost=sysCostprob)



bn11Cost = custom.fit(BNLayerOneCost, cptCost)
#graphviz.chart(bn11Cost,grid = TRUE)

graphviz.chart(bn11Cost, type = "barprob",scale = c(1.25, 1.5), grid = TRUE, bar.col = "darkgreen",strip.bg = "white")

##End...


sim = cpdist(bn11, nodes = "SysTT", n = 1000,evidence = ((MR1 == 0) & (MR2 == 0)))
simcost = cpdist(bn11Cost, nodes = "SysCost", n = 1000,evidence = ((MR1 == 0) & (MR2 == 0)))
hist(sim$SysTT)
typeof(sim$SysTT)
hist(list(sim$SysTT))
View(sim$SysTT)
simPure <- gsub(",", "", sim$SysTT)   # remove comma
simPure <- as.numeric(simPure)
hist(simPure,col="blue",freq = FALSE)
h <- hist(simPure, plot=FALSE)
h$density = h$counts/sum(h$counts) * 100
plot(h, main="Distribution of Salaries",
     xlab="Salary ($K)",
     ylab="Percent",
     col="blue",
     freq=FALSE)


data <- data.frame(
  type = c( rep("SysCost", 1000), rep("SysTT", 1000) ),
  value = c( simcost, sim )
)

# Represent it
data %>%
  ggplot( aes(x=value, fill=type)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  theme_ipsum() +
  labs(fill="")




hist(sim$SysTT)
View(sim)
#if prob having maintenance rate 1 being=1 (and not =0); 
#then, probability of system costs being =<1 will be
set.seed(1)
run11Costs<-replicate (1000,cpquery(bn11Cost, event =  (SysCost == 1) | (SysCost == 1), evidence = (MR1 == 0),n=10000))
#summary(run11Costs)
#hist(run11Costs)

#if prob having maintenance rate 1 being=1 (and not =0); 
#then, probability of system costs being =<1 will be
run11TT<-replicate (1000,cpquery(bn11, event =  (SysTT == 0) | (SysTT == 1), evidence =(MR1 == 0) ,n=10000))
#summary(run11TT)
#hist(run11TT)


#if prob having maintenance rate 1 being=1 (and not =0); 
#then, probability of system costs being =<1 will be
run12TT<-replicate (1000,cpquery(bn11, event =  (SysTT == 0) , evidence = (MR2 == 0),n=10000))
summary(run12TT)
hist(run12TT)

#https://fcurban.nl/eindhoven/
#https://r-graph-gallery.com/histogram_several_group.html
#https://stats.stackexchange.com/questions/234235/in-bnlearn-cpquery-gives-random-probablities
# Build dataset with different distributions
#https://www.youtube.com/watch?v=UOUrdDeQD3Y 
##See the above BN as a generative model..
##This plot shows if we take Mainteance rate equal 0, then prob of having system level costs<0,1 or travel time <=0,1
data <- data.frame(
  type = c( rep("Costs", 1000), rep("Travel Time", 1000) ),
  value = c( run11Costs, run11TT )
)

# Represent it
data %>%
  ggplot( aes(x=value, fill=type)) +
  geom_histogram( color="#e9ecef", alpha=0.6, position = 'identity') +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  theme_ipsum() +
  labs(fill="")


?cpquery
##############################Here

UniEdge<-list(unique(DFCondiProbTableLevels21$DepenVarLvel))
View(dfdata)
typeof(UniEdge)
UniEdge





#Variables: eta
#greeks = c(alpha='\u03b1', tau='\u03c4', sigma='\u03c3', sigmaSq='\u03c3\u00B2', beta='\u03b2', gamma='\u03b3')

BNLayerOneTime = model2network(paste0("[MR1][MR2][R12Avl|MR1][R13Avl|MR2][SysTT|R12Avl:R13Avl]",
                                  ""))
graphviz.plot(BNLayerOneTime, highlight = 
                list(nodes=nodes(BNLayerOneTime), fill="lightblue", col="black"))#layout = "dot")

g <- bnlearn::as.graphNEL(BNLayerOneTime) # use this to avoid printing of graphviz.plot
plot(g,  attrs=list(node = list(fillcolor = "lightgreen", fontsize=17,fontname="times-bold")))
#https://www.bnlearn.com/examples/graphviz-plot/

BNLayerOneTime11 = model2network(paste0("[Eta][Vteta][Eg13TAva|Vteta:Eta][Eg40TAva|Vteta:Eta][Eg50TAva|Vteta:Eta]",
                                        "[Subs4TAva|Eg13TAva:Eg40TAva:Eg50TAva]"))

graphviz.plot(BNLayerOneTime11, highlight = 
                list(nodes=nodes(BNLayerOneTime11), fill="lightblue", col="black"))#layout = "dot")


BNLayerOneTime12 = model2network(paste0("[Eta][Vteta][Eg13TAva|Vteta:Eta]"))

exceldata <- read.csv(file.choose(),sep = ";",2)                                                                             
View(exceldata)
dfdata = data.frame(exceldata)
UniEdge<-list(unique(dfdata$Edge))
View(dfdata)
typeof(UniEdge)


##Begin
setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-04/DiscretizeAlg1")
exceldata <- read.csv("2021-10-21-CondiProbTableEdgeTime1Alg1v01.csv",sep = ",",2)                                                                             
exceldata2 <- read.csv("2021-10-21-CondiProbTableEdgeTime2Alg1v01.csv",sep = ",",2)                                                                             
View(exceldata)
dfdata = data.frame(exceldata)
dfdata2 = data.frame(exceldata2)
posVal0=c(0,1,2)
posVal3=c(0,1,2,3)
posVal1=c(0,1,2,3,4)
Eta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Eta= posVal0))
Vteta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Vteta= posVal0))

UniEdge0<-(unique(dfdata$Edge))
UniEdge1<-(unique(dfdata2$Edge))
EdgeVarNamelist <- c()

for (edgeEle in UniEdge0) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|Vteta:Eta]",sep = ""),sep = "")
}

for (edgeEle in UniEdge1) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|Vteta:Eta]",sep = ""),sep = "")
}


EdgeVarNamelist
ModelOneString<-paste(EdgeVarNamelist, collapse = '')
ModelOneString<-paste("[Eta][Vteta]",ModelOneString,sep = "")
ModelOneString<-"[Eta][Vteta][Edge22TAva|Vteta:Eta][Edge63TAva|Vteta:Eta][Edge37TAva|Vteta:Eta][Edge50TAva|Vteta:Eta][Edge13TAva|Vteta:Eta][Edge40TAva|Vteta:Eta][Edge24TAva|Vteta:Eta][Edge43TAva|Vteta:Eta][Edge49TAva|Vteta:Eta][Edge58TAva|Vteta:Eta][Edge57TAva|Vteta:Eta][subsys5TAva|Edge22TAva:Edge24TAva][subsys6TAva|Edge37TAva:Edge49TAva:Edge43TAva][subsys8TAva|Edge58TAva][subsys9TAva|Edge63TAva:Edge57TAva]"

BNLayerOneTime0001 = model2network(ModelOneString)
graphviz.plot(BNLayerOneTime0001, highlight = 
                list(nodes=nodes(BNLayerOneTime0001), fill="lightblue", col="black"))#layout = "dot")

g <- bnlearn::as.graphNEL(BNLayerOneTime0001) 
plot(g,  attrs=list(node = list(fillcolor = "lightblue", fontsize=17,fontname="times-bold")))
(Eg13TAvaprob11)
Eg13TAvaprob11<-dfdata[dfdata$Edge==13,"ProbValue"]
Eg13TAvaprob<-array(Eg13TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge13TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg22TAvaprob11<-dfdata[dfdata$Edge==22,"ProbValue"]
Eg22TAvaprob<-array(Eg22TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge22TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg37TAvaprob11<-dfdata[dfdata$Edge==37,"ProbValue"]
Eg37TAvaprob<-array(Eg37TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge37TAva= posVal1, Vteta = posVal0,Eta = posVal0))



Eg50TAvaprob11<-dfdata[dfdata$Edge==50,"ProbValue"]
Eg50TAvaprob<-array(Eg50TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge50TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg63TAvaprob11<-dfdata[dfdata$Edge==63,"ProbValue"]
Eg63TAvaprob<-array(Eg63TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge63TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg24TAvaprob11<-dfdata2[dfdata2$Edge==24,"ProbValue"]
Eg24TAvaprob<-array(Eg24TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge24TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg40TAvaprob11<-dfdata2[dfdata2$Edge==40,"ProbValue"]
Eg40TAvaprob<-array(Eg40TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge40TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg43TAvaprob11<-dfdata2[dfdata2$Edge==43,"ProbValue"]
Eg43TAvaprob<-array(Eg43TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge43TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg49TAvaprob11<-dfdata2[dfdata2$Edge==49,"ProbValue"]
Eg49TAvaprob<-array(Eg49TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge49TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg57TAvaprob11<-dfdata2[dfdata2$Edge==57,"ProbValue"]
Eg57TAvaprob<-array(Eg57TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge57TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg58TAvaprob11<-dfdata2[dfdata2$Edge==58,"ProbValue"]
Eg58TAvaprob<-array(Eg58TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge58TAva= posVal1, Vteta = posVal0,Eta = posVal0))


cpt=list(Eta=Eta.prob,Vteta=Vteta.prob,Edge13TAva=Eg13TAvaprob,Edge22TAva=Eg22TAvaprob,
         Edge37TAva=Eg37TAvaprob,Edge50TAva=Eg50TAvaprob,Edge63TAva=Eg63TAvaprob,
         Edge24TAva=Eg24TAvaprob,Edge40TAva=Eg40TAvaprob,Edge43TAva=Eg43TAvaprob,
         Edge49TAva=Eg49TAvaprob,Edge57TAva=Eg57TAvaprob,Edge58TAva=Eg58TAvaprob)

cpt=list(Eta=Eta.prob,Vteta=Vteta.prob,Edge13TAva=Eg13TAvaprob,Edge22TAva=Eg22TAvaprob,
         Edge37TAva=Eg37TAvaprob,Edge50TAva=Eg50TAvaprob,Edge63TAva=Eg63TAvaprob,
         Edge24TAva=Eg24TAvaprob,Edge40TAva=Eg40TAvaprob,Edge43TAva=Eg43TAvaprob,
         Edge49TAva=Eg49TAvaprob,Edge57TAva=Eg57TAvaprob,Edge58TAva=Eg58TAvaprob)


bn11 = custom.fit(BNLayerOneTime0001, cpt)
graphviz.chart(bn11,grid = TRUE)

#End1


##TimmmmmeeeeeeeeeeeeeeBegin2
setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-04/DiscretizeAlg1")
exceldataold <- read.csv("2021-10-21-CondiProbTableEdgeTime1Alg1v01OldBr.csv",sep = ",",2)                                                                             
exceldatanew <- read.csv("2021-10-21-CondiProbTableEdgeTime1Alg1v01NewBr.csv",sep = ",",2)                                                                             
exceldata2old <- read.csv("2021-10-21-CondiProbTableEdgeTime2Alg1v01OldBr.csv",sep = ",",2)                                                                             
exceldata2new <- read.csv("2021-10-21-CondiProbTableEdgeTime2Alg1v01NewBr.csv",sep = ",",2)                                                                             
View(exceldata)
dfdataold = data.frame(exceldataold)
dfdatanew = data.frame(exceldatanew)
dfdata2old = data.frame(exceldata2old)
dfdata2new = data.frame(exceldata2new)

View(dfdataold)
View(dfdatanew)

setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-04/2021NovDiscretizeAlg3V01")
exceldatasubs <- read.csv("2021-10-21-CondiProbTableSubsystemTime1Alg1v09.csv",sep = ",",2)                                                                             
subsdfdata=data.frame(exceldatasubs)
View(subsdfdata)


posVal0=c(0,1,2)
posVal3=c(0,1,2,3)
posVal1=c(0,1,2,3,4)
Eta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Eta= posVal0))
Vteta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Vteta= posVal0))
newEta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(newEta= posVal0))
newVteta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(newVteta= posVal0))

UniEdge0old<-(unique(dfdataold$Edge))
UniEdge1old<-(unique(dfdata2old$Edge))
UniEdge0new<-(unique(dfdatanew$Edge))
UniEdge1new<-(unique(dfdata2new$Edge))

EdgeVarNamelist <- c()

for (edgeEle in UniEdge0old) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|Vteta:Eta]",sep = ""),sep = "")
}

for (edgeEle in UniEdge1old) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|Vteta:Eta]",sep = ""),sep = "")
}


for (edgeEle in UniEdge0new) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|newVteta:newEta]",sep = ""),sep = "")
}

for (edgeEle in UniEdge1new) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|newVteta:newEta]",sep = ""),sep = "")
}


Unisubsys<-(unique(subsdfdata$Subsystem))
pasteMine <- function(..., sep='') {
  paste(..., sep='', collapse='')
}

#eduni<-unique(subsdfdata[subsdfdata$Subsystem==4,"Edge1"])
for (subsEle in Unisubsys) {
  print(subsEle)
  edguni1<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge1"])
  edguni2<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge2"])
  edguni3<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge3"])
  if (edguni1>0){edge1Stringsubsys<-pasteMine("Edge",as.character(edguni1),"TAva")}
  else{edge1Stringsubsys<-""}
  if (edguni2>0){edge2Stringsubsys<-pasteMine(":Edge",as.character(edguni2),"TAva")}
  else{edge2Stringsubsys<-""}
  if (edguni3>0){edge3Stringsubsys<-pasteMine(":Edge",as.character(edguni3),"TAva")}
  else{edge3Stringsubsys<-""}
  
  EdgeVarNamelist<-pasteMine(EdgeVarNamelist,"[subsys",as.character(subsEle),"TAva|",edge1Stringsubsys,edge2Stringsubsys,edge3Stringsubsys,"]")
  #EdgeVarNamelist<-pasteMine(EdgeVarNamelist,edge1Stringsubsys,edge2Stringsubsys,edge3Stringsubsys,"]")
}



EdgeVarNamelist
ModelOneString<-paste(EdgeVarNamelist, collapse = '')
ModelOneString<-paste("[Eta][Vteta][newEta][newVteta]",ModelOneString,sep = "")
ModelOneString



BNLayerOneTime0001 = model2network(ModelOneString)
graphviz.plot(BNLayerOneTime0001, highlight = 
                list(nodes=nodes(BNLayerOneTime0001), fill="lightblue", col="black"))#layout = "dot")

g <- bnlearn::as.graphNEL(BNLayerOneTime0001) 
plot(g,  attrs=list(node = list(fillcolor = "lightblue", fontsize=17,fontname="times-bold")))


length(Eg13TAvaprob11)
length(Eg22TAvaprob11)
length(Eg37TAvaprob11)
length(Eg50TAvaprob11)
length(Eg63TAvaprob11)
length(Eg40TAvaprob11)
length(Eg49TAvaprob11)
length(Eg57TAvaprob11)
length(Eg58TAvaprob11)

Eg13TAvaprob11<-dfdatanew[dfdatanew$Edge==13,"ProbValue"]
Eg13TAvaprob<-array(Eg13TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge13TAva= posVal1, newVteta = posVal0,newEta = posVal0))

Eg22TAvaprob11<-dfdataold[dfdataold$Edge==22,"ProbValue"]
Eg22TAvaprob<-array(Eg22TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge22TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg37TAvaprob11<-dfdataold[dfdataold$Edge==37,"ProbValue"]
Eg37TAvaprob<-array(Eg37TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge37TAva= posVal1, Vteta = posVal0,Eta = posVal0))



Eg50TAvaprob11<-dfdatanew[dfdatanew$Edge==50,"ProbValue"]
Eg50TAvaprob<-array(Eg50TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge50TAva= posVal1, newVteta = posVal0,newEta = posVal0))


Eg63TAvaprob11<-dfdataold[dfdataold$Edge==63,"ProbValue"]
Eg63TAvaprob11
Eg63TAvaprob<-array(Eg63TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge63TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg24TAvaprob11<-dfdata2new[dfdata2new$Edge==24,"ProbValue"]
Eg24TAvaprob<-array(Eg24TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge24TAva= posVal1, newVteta = posVal0,newEta = posVal0))


Eg40TAvaprob11<-dfdata2old[dfdata2old$Edge==40,"ProbValue"]
Eg40TAvaprob<-array(Eg40TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge40TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg43TAvaprob11<-dfdata2new[dfdata2new$Edge==43,"ProbValue"]
Eg43TAvaprob<-array(Eg43TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge43TAva= posVal1, newVteta = posVal0,newEta = posVal0))

Eg49TAvaprob11<-dfdata2old[dfdata2old$Edge==49,"ProbValue"]
Eg49TAvaprob<-array(Eg49TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge49TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg57TAvaprob11<-dfdata2old[dfdata2old$Edge==57,"ProbValue"]
Eg57TAvaprob<-array(Eg57TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge57TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg58TAvaprob11<-dfdata2old[dfdata2old$Edge==58,"ProbValue"]
Eg58TAvaprob<-array(Eg58TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge58TAva= posVal1, Vteta = posVal0,Eta = posVal0))

subsys4TAvaprob11<-subsdfdata[subsdfdata$Subsystem==4,"ProbValue"]
subsys4TAvaprob<-array(subsys4TAvaprob11, dim = c(4, 5, 5,5),
                    dimnames = list(subsys4TAva= posVal3, Edge40TAva = posVal1,Edge13TAva = posVal1,Edge50TAva = posVal1))

subsys5TAvaprob11<-subsdfdata[subsdfdata$Subsystem==5,"ProbValue"]
subsys5TAvaprob<-array(subsys5TAvaprob11, dim = c(4, 5, 5),
                       dimnames = list(subsys5TAva= posVal3, Edge24TAva = posVal1,Edge22TAva = posVal1))

subsys6TAvaprob11<-subsdfdata[subsdfdata$Subsystem==6,"ProbValue"]
subsys6TAvaprob<-array(subsys6TAvaprob11, dim = c(4, 5, 5,5),
                       dimnames = list(subsys6TAva= posVal3, Edge43TAva = posVal1,Edge49TAva = posVal1,Edge37TAva = posVal1))

subsys8TAvaprob11<-subsdfdata[subsdfdata$Subsystem==8,"ProbValue"]
typeof(subsys8TAvaprob11)
subsys8TAvaprob11
typeof(Eg49TAvaprob11)
subsys8TAvaprob<-array(subsys8TAvaprob11, dim = c(4, 5),
                       dimnames = list(subsys8TAva= posVal3, Edge58TAva = posVal1))


subsys9TAvaprob11<-subsdfdata[subsdfdata$Subsystem==9,"ProbValue"]
subsys9TAvaprob<-array(subsys9TAvaprob11, dim = c(4, 5, 5),
                       dimnames = list(subsys9TAva= posVal3, Edge57TAva = posVal1,Edge63TAva = posVal1))

cpt=list(newEta=newEta.prob,newVteta=newVteta.prob,Eta=Eta.prob,Vteta=Vteta.prob,Edge13TAva=Eg13TAvaprob,Edge22TAva=Eg22TAvaprob,
         Edge37TAva=Eg37TAvaprob,Edge50TAva=Eg50TAvaprob,Edge63TAva=Eg63TAvaprob,
         Edge24TAva=Eg24TAvaprob,Edge40TAva=Eg40TAvaprob,Edge43TAva=Eg43TAvaprob,
         Edge49TAva=Eg49TAvaprob,Edge57TAva=Eg57TAvaprob,Edge58TAva=Eg58TAvaprob,
         subsys4TAva=subsys4TAvaprob,subsys5TAva=subsys5TAvaprob,subsys6TAva=subsys6TAvaprob,
         subsys8TAva=subsys8TAvaprob,subsys9TAva=subsys9TAvaprob)



bn11 = custom.fit(BNLayerOneTime0001, cpt)
graphviz.chart(bn11,grid = TRUE)

#TimeeeeeeeeeeeeeeeEnd2





##CooooossssssssssssssssstBegin2
setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-04/DiscretizeAlg1")
exceldataold <- read.csv("2021-10-21-CondiProbTableEdgeCost1Alg1v03.csv",sep = ",",2)                                                                             
exceldatanew <- read.csv("2021-10-21-CondiProbTableEdgeCost1Alg1v03.csv",sep = ",",2)                                                                             
exceldata2old <- read.csv("2021-10-21-CondiProbTableEdgeCost2Alg1v03OldBr.csv",sep = ",",2)                                                                             
exceldata2new <- read.csv("2021-10-21-CondiProbTableEdgeCost2Alg1v03OldBr.csv",sep = ",",2)                                                                             
View(exceldata)
dfdataold = data.frame(exceldataold)
dfdatanew = data.frame(exceldatanew)
dfdata2old = data.frame(exceldata2old)
dfdata2new = data.frame(exceldata2new)

View(dfdataold)
View(dfdatanew)

setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-04/2021NovDiscretizeAlg3V01")
exceldatasubs <- read.csv("2021-10-21-CondiProbTableSubsystemCost1Alg1v03.csv",sep = ",",2)                                                                             
subsdfdata=data.frame(exceldatasubs)
View(subsdfdata)


posVal0=c(0,1,2)
posVal2=c(0,1)
posVal3=c(0,1,2,3)
posVal1=c(0,1,2,3,4)
Etacost.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Etacost= posVal0))
Vtetacost.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Vtetacost= posVal0))
newEtacost.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(newEtacost= posVal0))
newVtetacost.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(newVtetacost= posVal0))
costfaca.prob = array(c(0.2, 0.8), dim = 2, dimnames = list(costfaca= posVal2))
costfacb.prob = array(c(0.2, 0.8), dim = 2, dimnames = list(costfacb= posVal2))
newcostfaca.prob = array(c(0.2, 0.8), dim = 2, dimnames = list(newcostfaca= posVal2))
newcostfacb.prob = array(c(0.2, 0.8), dim = 2, dimnames = list(newcostfacb= posVal2))




UniEdge0old<-c(22,63,37)#(unique(dfdataold$Edge))
UniEdge1old<-c(40,49,57,58)#(unique(dfdata2old$Edge))
UniEdge0new<-c(13,50)#(unique(dfdatanew$Edge))
UniEdge1new<-c(24,43)#(unique(dfdata2new$Edge))

UniEdge0old
UniEdge1new
UniEdge0new
UniEdge1new
EdgeVarNamelist <- c()

for (edgeEle in UniEdge0old) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"Cost|Vtetacost:Etacost:costfaca:costfacb]",sep = ""),sep = "")
}

for (edgeEle in UniEdge1old) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"Cost|Vtetacost:Etacost:costfaca:costfacb]",sep = ""),sep = "")
}


for (edgeEle in UniEdge0new) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"Cost|newVtetacost:newEtacost:newcostfaca:newcostfacb]",sep = ""),sep = "")
}

for (edgeEle in UniEdge1new) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"Cost|newVtetacost:newEtacost:newcostfaca:newcostfacb]",sep = ""),sep = "")
}


Unisubsys<-(unique(subsdfdata$Subsystem))
pasteMine <- function(..., sep='') {
  paste(..., sep='', collapse='')
}

#eduni<-unique(subsdfdata[subsdfdata$Subsystem==4,"Edge1"])
for (subsEle in Unisubsys) {
  print(subsEle)
  edguni1<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge1"])
  edguni2<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge2"])
  edguni3<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge3"])
  if (edguni1>0){edge1Stringsubsys<-pasteMine("Edge",as.character(edguni1),"Cost")}
  else{edge1Stringsubsys<-""}
  if (edguni2>0){edge2Stringsubsys<-pasteMine(":Edge",as.character(edguni2),"Cost")}
  else{edge2Stringsubsys<-""}
  if (edguni3>0){edge3Stringsubsys<-pasteMine(":Edge",as.character(edguni3),"Cost")}
  else{edge3Stringsubsys<-""}
  
  EdgeVarNamelist<-pasteMine(EdgeVarNamelist,"[subsys",as.character(subsEle),"Cost|",edge1Stringsubsys,edge2Stringsubsys,edge3Stringsubsys,"]")
  #EdgeVarNamelist<-pasteMine(EdgeVarNamelist,edge1Stringsubsys,edge2Stringsubsys,edge3Stringsubsys,"]")
}



EdgeVarNamelist
ModelOneString<-paste(EdgeVarNamelist, collapse = '')
ModelOneString<-paste("[costfaca][costfacb][newcostfaca][newcostfacb][Etacost][Vtetacost][newEtacost][newVtetacost]",ModelOneString,sep = "")
ModelOneString



BNLayerOneCost0001 = model2network(ModelOneString)
graphviz.plot(BNLayerOneCost0001, highlight = 
                list(nodes=nodes(BNLayerOneTime0001), fill="lightblue", col="black"))#layout = "dot")

g <- bnlearn::as.graphNEL(BNLayerOneCost0001) 
plot(g,  attrs=list(node = list(fillcolor = "lightblue", fontsize=17,fontname="times-bold")))


Eg13Costprob11
Eg22Costprob11
length(Eg13Costprob11)
length(Eg22Costprob11)
length(Eg37Costprob11)
length(Eg50Costprob11)
length(Eg63Costprob11)
length(Eg40Costprob11)
length(Eg49Costprob11)
length(Eg57Costprob11)
length(Eg58Costprob11)

Eg13Costprob11<-dfdatanew[dfdatanew$Edge==13,"ProbValue"]
Eg13Costprob<-array(Eg13Costprob11, dim = c(5, 2, 2,3,3),
                    dimnames = list(Edge13Cost= posVal1, newcostfacb=posVal2,newcostfaca=posVal2,newVtetacost = posVal0,newEtacost = posVal0))

Eg22Costprob11<-dfdataold[dfdataold$Edge==22,"ProbValue"]
Eg22Costprob<-array(Eg22Costprob11, dim = c(5, 2, 2,3,3),
                    dimnames = list(Edge22Cost= posVal1, costfacb=posVal2,costfaca=posVal2,Vtetacost = posVal0,Etacost = posVal0))


Eg37Costprob11<-dfdataold[dfdataold$Edge==37,"ProbValue"]
Eg37Costprob<-array(Eg37Costprob11, dim = c(5, 2, 2,3,3),
                    dimnames = list(Edge37Cost= posVal1, costfacb=posVal2,costfaca=posVal2,Vtetacost = posVal0,Etacost = posVal0))



Eg50Costprob11<-dfdatanew[dfdatanew$Edge==50,"ProbValue"]
Eg50Costprob<-array(Eg50Costprob11, dim = c(5, 2, 2,3,3),
                    dimnames = list(Edge50Cost= posVal1, newcostfacb=posVal2,newcostfaca=posVal2,newVtetacost = posVal0,newEtacost = posVal0))


Eg63Costprob11<-dfdataold[dfdataold$Edge==63,"ProbValue"]
Eg63Costprob11
Eg63Costprob<-array(Eg63Costprob11, dim = c(5, 2, 2,3,3),
                    dimnames = list(Edge63Cost= posVal1, costfacb=posVal2,costfaca=posVal2,Vtetacost = posVal0,Etacost = posVal0))


Eg24Costprob11<-dfdata2new[dfdata2new$Edge==24,"ProbValue"]
Eg24Costprob<-array(Eg24Costprob11, dim = c(5, 2,2,3,3),
                    dimnames = list(Edge24Cost= posVal1, newcostfacb=posVal2,newcostfaca=posVal2,newVtetacost = posVal0,newEtacost = posVal0))


Eg40Costprob11<-dfdata2old[dfdata2old$Edge==40,"ProbValue"]
Eg40Costprob<-array(Eg40Costprob11, dim = c(5, 2,2,3,3),
                    dimnames = list(Edge40Cost= posVal1, costfacb=posVal2,costfaca=posVal2,Vtetacost = posVal0,Etacost = posVal0))


Eg43Costprob11<-dfdata2new[dfdata2new$Edge==43,"ProbValue"]
Eg43Costprob<-array(Eg43Costprob11, dim = c(5, 2,2,3,3),
                    dimnames = list(Edge43Cost= posVal1, newcostfacb=posVal2,newcostfaca=posVal2,newVtetacost = posVal0,newEtacost = posVal0))

Eg49Costprob11<-dfdata2old[dfdata2old$Edge==49,"ProbValue"]
Eg49Costprob<-array(Eg49Costprob11, dim = c(5, 2,2,3,3),
                    dimnames = list(Edge49Cost= posVal1, costfacb=posVal2,costfaca=posVal2,Vtetacost = posVal0,Etacost = posVal0))

Eg57Costprob11<-dfdata2old[dfdata2old$Edge==57,"ProbValue"]
Eg57Costprob<-array(Eg57Costprob11, dim = c(5, 2,2,3,3),
                    dimnames = list(Edge57Cost= posVal1, costfacb=posVal2,costfaca=posVal2,Vtetacost = posVal0,Etacost = posVal0))

Eg58Costprob11<-dfdata2old[dfdata2old$Edge==58,"ProbValue"]
Eg58Costprob<-array(Eg58Costprob11, dim = c(5, 2,2,3,3),
                    dimnames = list(Edge58Cost= posVal1, costfacb=posVal2,costfaca=posVal2,Vtetacost = posVal0,Etacost = posVal0))

subsys4Costprob11<-subsdfdata[subsdfdata$Subsystem==4,"ProbValue"]
subsys4Costprob<-array(subsys4Costprob11, dim = c(4, 5, 5,5),
                       dimnames = list(subsys4Cost= posVal3, Edge40Cost = posVal1,Edge13Cost = posVal1,Edge50Cost = posVal1))

subsys5Costprob11<-subsdfdata[subsdfdata$Subsystem==5,"ProbValue"]
subsys5Costprob<-array(subsys5Costprob11, dim = c(4, 5, 5),
                       dimnames = list(subsys5Cost= posVal3, Edge24Cost = posVal1,Edge22Cost = posVal1))

subsys6Costprob11<-subsdfdata[subsdfdata$Subsystem==6,"ProbValue"]
subsys6Costprob<-array(subsys6Costprob11, dim = c(4, 5, 5,5),
                       dimnames = list(subsys6Cost= posVal3, Edge43Cost = posVal1,Edge49Cost = posVal1,Edge37Cost = posVal1))

subsys8Costprob11<-subsdfdata[subsdfdata$Subsystem==8,"ProbValue"]
typeof(subsys8Costprob11)
subsys8Costprob11
typeof(Eg49Costprob11)
subsys8Costprob<-array(subsys8Costprob11, dim = c(4, 5),
                       dimnames = list(subsys8Cost= posVal3, Edge58Cost = posVal1))


subsys9Costprob11<-subsdfdata[subsdfdata$Subsystem==9,"ProbValue"]
subsys9Costprob<-array(subsys9Costprob11, dim = c(4, 5, 5),
                       dimnames = list(subsys9Cost= posVal3, Edge57Cost = posVal1,Edge63Cost = posVal1))

cpt=list(costfacb=costfacb.prob,costfaca=costfaca.prob,newcostfacb=newcostfacb.prob,newcostfaca=newcostfaca.prob,newEtacost=newEtacost.prob,newVtetacost=newVtetacost.prob,Etacost=Etacost.prob,Vtetacost=Vtetacost.prob,Edge13Cost=Eg13Costprob,Edge22Cost=Eg22Costprob,
         Edge37Cost=Eg37Costprob,Edge50Cost=Eg50Costprob,Edge63Cost=Eg63Costprob,
         Edge24Cost=Eg24Costprob,Edge40Cost=Eg40Costprob,Edge43Cost=Eg43Costprob,
         Edge49Cost=Eg49Costprob,Edge57Cost=Eg57Costprob,Edge58Cost=Eg58Costprob,
         subsys4Cost=subsys4Costprob,subsys5Cost=subsys5Costprob,subsys6Cost=subsys6Costprob,
         subsys8Cost=subsys8Costprob,subsys9Cost=subsys9Costprob)



bn11 = custom.fit(BNLayerOneCost0001, cpt)
graphviz.chart(bn11,grid = TRUE)

#CooostttttttEnd2










##TimmmmmeeeeeeeeeeeeeeBegin2
setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-04/DiscretizeAlg1")
exceldataold <- read.csv("2021-10-21-CondiProbTableEdgeTime1Alg1v01OldBr.csv",sep = ",",2)                                                                             
exceldatanew <- read.csv("2021-10-21-CondiProbTableEdgeTime1Alg1v01NewBr.csv",sep = ",",2)                                                                             
exceldata2old <- read.csv("2021-10-21-CondiProbTableEdgeTime2Alg1v01OldBr.csv",sep = ",",2)                                                                             
exceldata2new <- read.csv("2021-10-21-CondiProbTableEdgeTime2Alg1v01NewBr.csv",sep = ",",2)                                                                             
View(exceldata)
dfdataold = data.frame(exceldataold)
dfdatanew = data.frame(exceldatanew)
dfdata2old = data.frame(exceldata2old)
dfdata2new = data.frame(exceldata2new)

View(dfdataold)
View(dfdatanew)

setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-04/2021NovDiscretizeAlg3V01")
exceldatasubs <- read.csv("2021-10-21-CondiProbTableSubsystemTime1Alg1v09.csv",sep = ",",2)                                                                             
subsdfdata=data.frame(exceldatasubs)
View(subsdfdata)


posVal0=c(0,1,2)
posVal3=c(0,1,2,3)
posVal1=c(0,1,2,3,4)
Eta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Eta= posVal0))
Vteta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Vteta= posVal0))
newEta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(newEta= posVal0))
newVteta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(newVteta= posVal0))

UniEdge0old<-(unique(dfdataold$Edge))
UniEdge1old<-(unique(dfdata2old$Edge))
UniEdge0new<-(unique(dfdatanew$Edge))
UniEdge1new<-(unique(dfdata2new$Edge))

EdgeVarNamelist <- c()

for (edgeEle in UniEdge0old) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|Vteta:Eta]",sep = ""),sep = "")
}

for (edgeEle in UniEdge1old) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|Vteta:Eta]",sep = ""),sep = "")
}


for (edgeEle in UniEdge0new) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|newVteta:newEta]",sep = ""),sep = "")
}

for (edgeEle in UniEdge1new) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|newVteta:newEta]",sep = ""),sep = "")
}


Unisubsys<-(unique(subsdfdata$Subsystem))
pasteMine <- function(..., sep='') {
  paste(..., sep='', collapse='')
}

#eduni<-unique(subsdfdata[subsdfdata$Subsystem==4,"Edge1"])
for (subsEle in Unisubsys) {
  print(subsEle)
  edguni1<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge1"])
  edguni2<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge2"])
  edguni3<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge3"])
  if (edguni1>0){edge1Stringsubsys<-pasteMine("Edge",as.character(edguni1),"TAva")}
  else{edge1Stringsubsys<-""}
  if (edguni2>0){edge2Stringsubsys<-pasteMine(":Edge",as.character(edguni2),"TAva")}
  else{edge2Stringsubsys<-""}
  if (edguni3>0){edge3Stringsubsys<-pasteMine(":Edge",as.character(edguni3),"TAva")}
  else{edge3Stringsubsys<-""}
  
  EdgeVarNamelist<-pasteMine(EdgeVarNamelist,"[subsys",as.character(subsEle),"TAva|",edge1Stringsubsys,edge2Stringsubsys,edge3Stringsubsys,"]")
  #EdgeVarNamelist<-pasteMine(EdgeVarNamelist,edge1Stringsubsys,edge2Stringsubsys,edge3Stringsubsys,"]")
}



EdgeVarNamelist
ModelOneString<-paste(EdgeVarNamelist, collapse = '')
ModelOneString<-paste("[Eta][Vteta][newEta][newVteta]",ModelOneString,sep = "")
ModelOneString



BNLayerOneTime0001 = model2network(ModelOneString)
graphviz.plot(BNLayerOneTime0001, highlight = 
                list(nodes=nodes(BNLayerOneTime0001), fill="lightblue", col="black"))#layout = "dot")

g <- bnlearn::as.graphNEL(BNLayerOneTime0001) 
plot(g,  attrs=list(node = list(fillcolor = "lightblue", fontsize=17,fontname="times-bold")))


length(Eg13TAvaprob11)
length(Eg22TAvaprob11)
length(Eg37TAvaprob11)
length(Eg50TAvaprob11)
length(Eg63TAvaprob11)
length(Eg40TAvaprob11)
length(Eg49TAvaprob11)
length(Eg57TAvaprob11)
length(Eg58TAvaprob11)

Eg13TAvaprob11<-dfdata[dfdatanew$Edge==13,"ProbValue"]
Eg13TAvaprob<-array(Eg13TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge13TAva= posVal1, newVteta = posVal0,newEta = posVal0))

Eg22TAvaprob11<-dfdata[dfdataold$Edge==22,"ProbValue"]
Eg22TAvaprob<-array(Eg22TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge22TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg37TAvaprob11<-dfdata[dfdataold$Edge==37,"ProbValue"]
Eg37TAvaprob<-array(Eg37TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge37TAva= posVal1, Vteta = posVal0,Eta = posVal0))



Eg50TAvaprob11<-dfdata[dfdatanew$Edge==50,"ProbValue"]
Eg50TAvaprob<-array(Eg50TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge50TAva= posVal1, newVteta = posVal0,newEta = posVal0))


Eg63TAvaprob11<-dfdata[dfdataold$Edge==63,"ProbValue"]
Eg63TAvaprob11
Eg63TAvaprob<-array(Eg63TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge63TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg24TAvaprob11<-dfdata2[dfdata2new$Edge==24,"ProbValue"]
Eg24TAvaprob<-array(Eg24TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge24TAva= posVal1, newVteta = posVal0,newEta = posVal0))


Eg40TAvaprob11<-dfdata2[dfdata2old$Edge==40,"ProbValue"]
Eg40TAvaprob<-array(Eg40TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge40TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg43TAvaprob11<-dfdata2[dfdata2new$Edge==43,"ProbValue"]
Eg43TAvaprob<-array(Eg43TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge43TAva= posVal1, newVteta = posVal0,newEta = posVal0))

Eg49TAvaprob11<-dfdata2[dfdata2old$Edge==49,"ProbValue"]
Eg49TAvaprob<-array(Eg49TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge49TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg57TAvaprob11<-dfdata2[dfdata2old$Edge==57,"ProbValue"]
Eg57TAvaprob<-array(Eg57TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge57TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg58TAvaprob11<-dfdata2[dfdata2old$Edge==58,"ProbValue"]
Eg58TAvaprob<-array(Eg58TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge58TAva= posVal1, Vteta = posVal0,Eta = posVal0))

subsys4TAvaprob11<-subsdfdata[subsdfdata$Subsystem==4,"ProbValue"]
subsys4TAvaprob<-array(subsys4TAvaprob11, dim = c(4, 5, 5,5),
                       dimnames = list(subsys4TAva= posVal3, Edge40TAva = posVal1,Edge13TAva = posVal1,Edge50TAva = posVal1))

subsys5TAvaprob11<-subsdfdata[subsdfdata$Subsystem==5,"ProbValue"]
subsys5TAvaprob<-array(subsys5TAvaprob11, dim = c(4, 5, 5),
                       dimnames = list(subsys5TAva= posVal3, Edge24TAva = posVal1,Edge22TAva = posVal1))

subsys6TAvaprob11<-subsdfdata[subsdfdata$Subsystem==6,"ProbValue"]
subsys6TAvaprob<-array(subsys6TAvaprob11, dim = c(4, 5, 5,5),
                       dimnames = list(subsys6TAva= posVal3, Edge43TAva = posVal1,Edge49TAva = posVal1,Edge37TAva = posVal1))

subsys8TAvaprob11<-subsdfdata[subsdfdata$Subsystem==8,"ProbValue"]
typeof(subsys8TAvaprob11)
subsys8TAvaprob11
typeof(Eg49TAvaprob11)
subsys8TAvaprob<-array(subsys8TAvaprob11, dim = c(4, 5),
                       dimnames = list(subsys8TAva= posVal3, Edge58TAva = posVal1))


subsys9TAvaprob11<-subsdfdata[subsdfdata$Subsystem==9,"ProbValue"]
subsys9TAvaprob<-array(subsys9TAvaprob11, dim = c(4, 5, 5),
                       dimnames = list(subsys9TAva= posVal3, Edge57TAva = posVal1,Edge63TAva = posVal1))

cpt=list(newEta=newEta.prob,newVteta=newVteta.prob,Eta=Eta.prob,Vteta=Vteta.prob,Edge13TAva=Eg13TAvaprob,Edge22TAva=Eg22TAvaprob,
         Edge37TAva=Eg37TAvaprob,Edge50TAva=Eg50TAvaprob,Edge63TAva=Eg63TAvaprob,
         Edge24TAva=Eg24TAvaprob,Edge40TAva=Eg40TAvaprob,Edge43TAva=Eg43TAvaprob,
         Edge49TAva=Eg49TAvaprob,Edge57TAva=Eg57TAvaprob,Edge58TAva=Eg58TAvaprob,
         subsys4TAva=subsys4TAvaprob,subsys5TAva=subsys5TAvaprob,subsys6TAva=subsys6TAvaprob,
         subsys8TAva=subsys8TAvaprob,subsys9TAva=subsys9TAvaprob)



bn11 = custom.fit(BNLayerOneTime0001, cpt)
graphviz.chart(bn11,grid = TRUE)

#TimeeeeeeeeeeeeeeeEnd2









ModelOneString<-"[Eta][Vteta][Edge22TAva|Vteta:Eta][Edge63TAva|Vteta:Eta][Edge37TAva|Vteta:Eta][Edge50TAva|Vteta:Eta][Edge13TAva|Vteta:Eta][Edge40TAva|Vteta:Eta][Edge24TAva|Vteta:Eta][Edge43TAva|Vteta:Eta][Edge49TAva|Vteta:Eta][Edge58TAva|Vteta:Eta][Edge57TAva|Vteta:Eta][subsys8TAva|Edge58TAva]"
BNLayerOneTime0001 = model2network(ModelOneString)

cpt=list(Eta=Eta.prob,Vteta=Vteta.prob,Edge13TAva=Eg13TAvaprob,Edge22TAva=Eg22TAvaprob,
         Edge37TAva=Eg37TAvaprob,Edge50TAva=Eg50TAvaprob,Edge63TAva=Eg63TAvaprob,
         Edge24TAva=Eg24TAvaprob,Edge40TAva=Eg40TAvaprob,Edge43TAva=Eg43TAvaprob,
         Edge49TAva=Eg49TAvaprob,Edge57TAva=Eg57TAvaprob,Edge58TAva=Eg58TAvaprob,
         subsys8TAva=subsys8TAvaprob)




















#Begin



exceldatasubs <- read.csv(file.choose(),sep = ",",2)                                                                             
subsdfdata=data.frame(exceldatasubs)
View(subsdfdata)
UniEdge
posVal0=c(0,1,2)
posVal1=c(0,1,2,3,4)
Eta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Eta= posVal0))
Vteta.prob = array(c(0.1,0.1, 0.8), dim = 3, dimnames = list(Vteta= posVal0))

UniEdge0<-(unique(dfdata$Edge))
UniEdge1<-(unique(dfdata2$Edge))
EdgeVarNamelist <- c()

for (edgeEle in UniEdge0) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|Vteta:Eta]",sep = ""),sep = "")
}
for (edgeEle in UniEdge1) {
  print(edgeEle)
  EdgeVarNamelist<-paste(EdgeVarNamelist,paste("[Edge",as.character(edgeEle),"TAva|Vteta:Eta]",sep = ""),sep = "")
}


Unisubsys<-(unique(subsdfdata$Subsystem))
pasteMine <- function(..., sep='') {
  paste(..., sep='', collapse='')
}

#eduni<-unique(subsdfdata[subsdfdata$Subsystem==4,"Edge1"])
for (subsEle in Unisubsys) {
  print(subsEle)
  edguni1<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge1"])
  edguni2<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge2"])
  edguni3<-unique(subsdfdata[subsdfdata$Subsystem==subsEle,"Edge3"])
  if (edguni1>0){edge1Stringsubsys<-pasteMine("Edge",as.character(edguni1),"TAva")}
  else{edge1Stringsubsys<-""}
  if (edguni2>0){edge2Stringsubsys<-pasteMine(":Edge",as.character(edguni2),"TAva")}
  else{edge2Stringsubsys<-""}
  if (edguni3>0){edge3Stringsubsys<-pasteMine(":Edge",as.character(edguni3),"TAva")}
  else{edge3Stringsubsys<-""}

  EdgeVarNamelist<-pasteMine(EdgeVarNamelist,"[subsys",as.character(subsEle),"TAva|",edge1Stringsubsys,edge2Stringsubsys,edge3Stringsubsys,"]")
  #EdgeVarNamelist<-pasteMine(EdgeVarNamelist,edge1Stringsubsys,edge2Stringsubsys,edge3Stringsubsys,"]")
  }



EdgeVarNamelist
ModelOneString<-paste(EdgeVarNamelist, collapse = '')
ModelOneString<-paste("[Eta][Vteta]",ModelOneString,sep = "")
ModelOneString

BNLayerOneTime0001 = model2network(ModelOneString)
graphviz.plot(BNLayerOneTime0001, highlight = 
                list(nodes=nodes(BNLayerOneTime0001), fill="lightblue", col="black"))#layout = "dot")

g <- bnlearn::as.graphNEL(BNLayerOneTime0001) 
plot(g,  attrs=list(node = list(fillcolor = "lightblue", fontsize=17,fontname="times-bold")))

Eg13TAvaprob11<-dfdata[dfdata$Edge==13,"ProbValue"]
Eg13TAvaprob<-array(Eg13TAvaprob11, dim = c(5, 3, 3),
     dimnames = list(Edge13TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg22TAvaprob11<-dfdata[dfdata$Edge==22,"ProbValue"]
Eg22TAvaprob<-array(Eg22TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge22TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg37TAvaprob11<-dfdata[dfdata$Edge==37,"ProbValue"]
Eg37TAvaprob<-array(Eg37TAvaprob11, dim = c(5, 3, 3),
    dimnames = list(Edge37TAva= posVal1, Vteta = posVal0,Eta = posVal0))



Eg50TAvaprob11<-dfdata[dfdata$Edge==50,"ProbValue"]
Eg50TAvaprob<-array(Eg50TAvaprob11, dim = c(5, 3, 3),
    dimnames = list(Edge50TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg63TAvaprob11<-dfdata[dfdata$Edge==63,"ProbValue"]
Eg63TAvaprob<-array(Eg63TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge63TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg24TAvaprob11<-dfdata2[dfdata2$Edge==24,"ProbValue"]
Eg24TAvaprob<-array(Eg24TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge24TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg40TAvaprob11<-dfdata2[dfdata2$Edge==40,"ProbValue"]
Eg40TAvaprob<-array(Eg40TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge40TAva= posVal1, Vteta = posVal0,Eta = posVal0))


Eg43TAvaprob11<-dfdata2[dfdata2$Edge==43,"ProbValue"]
Eg43TAvaprob<-array(Eg43TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge43TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg49TAvaprob11<-dfdata2[dfdata2$Edge==49,"ProbValue"]
Eg49TAvaprob<-array(Eg49TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge49TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg57TAvaprob11<-dfdata2[dfdata2$Edge==57,"ProbValue"]
Eg57TAvaprob<-array(Eg57TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge57TAva= posVal1, Vteta = posVal0,Eta = posVal0))

Eg58TAvaprob11<-dfdata2[dfdata2$Edge==58,"ProbValue"]
Eg58TAvaprob<-array(Eg58TAvaprob11, dim = c(5, 3, 3),
                    dimnames = list(Edge58TAva= posVal1, Vteta = posVal0,Eta = posVal0))



subsys4TAvaprob11<-subsdfdata[subsdfdata$Subsystem==4,"ProbValue"]
subsys4TAvaprob<-array(subsys4TAvaprob11, dim = c(5, 5, 5,5),
                    dimnames = list(Edge58TAva= posVal1, Vteta = posVal0,Eta = posVal0))


cpt=list(Eta=Eta.prob,Vteta=Vteta.prob,Edge13TAva=Eg13TAvaprob,Edge22TAva=Eg22TAvaprob,
         Edge37TAva=Eg37TAvaprob,Edge50TAva=Eg50TAvaprob,Edge63TAva=Eg63TAvaprob,
         Edge24TAva=Eg24TAvaprob,Edge40TAva=Eg40TAvaprob,Edge43TAva=Eg43TAvaprob,
         Edge49TAva=Eg49TAvaprob,Edge57TAva=Eg57TAvaprob,Edge58TAva=Eg58TAvaprob)


bn11 = custom.fit(BNLayerOneTime0001, cpt)
graphviz.chart(bn11,grid = TRUE)
###End





#View(Eg13TAva.prob11)
#Eg13TAva.prob1111<-dfdata[dfdata$Edge==50,]
#View(Eg13TAva.prob1111)

cpt=list(Eta=Eta.prob,Vteta=Vteta.prob,Eg13TAva=Eg13TAva.prob)
bn11 = custom.fit(BNLayerOneTime12, cpt)
bn.fit.barchart(BNLayerOneTime12) 
bn.fit.dotplot(bn11)
graphviz.chart(bn11,grid = TRUE)


#do separate and then altogether time the other algorithms;










posVal=c(0,1,2,3,4,5)

Eta.prob = array(c(0.01,001, 0.98), dim = 3, dimnames = list(Eta= posVal0))
Vteta.prob = array(c(0.01,001, 0.98), dim = 3, dimnames = list(Vteta= posVal0))









asia.dag = model2network("[A][S][T|A][L|S][B|S][D|B:E][E|T:L][X|E]")
lv = c("yes", "no")
A.prob = array(c(0.01, 0.99), dim = 2, dimnames = list(A = lv))
S.prob = array(c(0.01, 0.99), dim = 2, dimnames = list(A = lv))
T.prob = array(c(0.05, 0.95, 0.01, 0.99), dim = c(2, 2),
               dimnames = list(T = lv, A = lv))
L.prob = array(c(0.1, 0.9, 0.01, 0.99), dim = c(2, 2),
               dimnames = list(L = lv, S = lv))
B.prob = array(c(0.6, 0.4, 0.3, 0.7), dim = c(2, 2),
               dimnames = list(B = lv, S = lv))
D.prob = array(c(0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.1, 0.9), dim = c(2, 2, 2),
               dimnames = list(D = lv, B = lv, E = lv))
E.prob = array(c(1, 0, 1, 0, 1, 0, 0, 1), dim = c(2, 2, 2),
               dimnames = list(E = lv, T = lv, L = lv))
X.prob = array(c(0.98, 0.02, 0.05, 0.95), dim = c(2, 2),
               dimnames = list(X = lv, E = lv))
cpt = list(A = A.prob, S = S.prob, T = T.prob, L = L.prob, B = B.prob,
           D = D.prob, E = E.prob, X = X.prob)
bn = custom.fit(asia.dag, cpt)

B.prob








BNLayerOne = model2network(paste0("[Eta][Vteta][Ca][Cb][Edg13TAva|Vteta:Eta]
[Edg22TAva|Vteta:Eta][Edg37TAva|Vteta:Eta]
                 [Edg50TAvail|Vteta:Eta]"))















rats.dag = model2network("[greeks['tau']][DRUG|greeks['tau']][WL1|DRUG][WL2|WL1:DRUG]")


asia.dag = model2network("[a[label=<&#945;>]][S][T|a[label=<&#945;>])][L|S][B|S][D|B:E][E|T:L][X|E]")
lv = c("yes", "no")
A.prob = array(c(0.01, 0.99), dim = 2, dimnames = list(A = lv))
S.prob = array(c(0.01, 0.99), dim = 2, dimnames = list(A = lv))
T.prob = array(c(0.05, 0.95, 0.01, 0.99), dim = c(2, 2),
               dimnames = list(T = lv, A = lv))
L.prob = array(c(0.1, 0.9, 0.01, 0.99), dim = c(2, 2),
               dimnames = list(L = lv, S = lv))
B.prob = array(c(0.6, 0.4, 0.3, 0.7), dim = c(2, 2),
               dimnames = list(B = lv, S = lv))
D.prob = array(c(0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.1, 0.9), dim = c(2, 2, 2),
               dimnames = list(D = lv, B = lv, E = lv))
E.prob = array(c(1, 0, 1, 0, 1, 0, 0, 1), dim = c(2, 2, 2),
               dimnames = list(E = lv, T = lv, L = lv))
X.prob = array(c(0.98, 0.02, 0.05, 0.95), dim = c(2, 2),
               dimnames = list(X = lv, E = lv))
cpt = list(A = A.prob, S = S.prob, T = T.prob, L = L.prob, B = B.prob,
           D = D.prob, E = E.prob, X = X.prob)
bn = custom.fit(asia.dag, cpt)













tree = model2network("[expression(theta)][B|expression(theta)][C|$\eta$][D|$\eta$][E|B][F|B][G|C][H|C][I|C]")
graphviz.plot(asia.dag, layout = "dot")

graphviz.plot(tree, highlight = list(nodes = "B",col = "tomato", fill = "orange"))


survey.dag = model2network("[A][S][E|A:S][O|E][R|E][T|O:R]")
hlight = list(nodes = c("E", "O"),
              arcs = c("E", "O"),
              col = "grey",
              textCol = "grey")
pp = graphviz.plot(survey.dag,
                   highlight = hlight)


nodeRenderInfo(pp) =
  list(col =
         c("S" = "black", "E" = "black",
           "R" = "black"),
       textCol =
         c("S" = "black", "E" = "black",
           "R" = "black"),
       fill = c("E" = "grey"))

renderGraph(pp)

#check inputs and outputs of alg1, and then, discretize for n_discretize items
#inputs of simulation and its outputs...

setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-02-FinalPythonCodeResults/20210820-Alg3Results/")
HighDensityAlpha<-0.9
NumIntervalDiscretise<-5
setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-02-FinalPythonCodeResults/20210820-Alg3Results/Algorithm2-MainRun-August30")
Alg2pd2OnlyAsset<-read.table("2021-08-21-pd2OnlyAsset.csv",header = TRUE,sep = ",")
Alg2subselpd1AssetWiSubset<-read.table("2021-08-21-All-subselpd1AssetWiSubset.csv",header = TRUE,sep = ",")
typeof(Alg2pd2OnlyAsset)
Alg2pd2OnlyAssetDF<-as.data.frame(Alg2pd2OnlyAsset)
DF<-Alg2pd2OnlyAsset
AllEges<-unique(Alg2pd2OnlyAsset$Edge)
DFWorkingSubet<-Alg2pd2OnlyAsset[Alg2pd2OnlyAsset$Edge == edge, ]  
for (edge in AllEges) {
  print(edge)
  Intervals<-hdr(DFWorkingSubet$TotalAvail,prob=0.7)
  UpperInt<-ifelse(Intervals$hdr[2]>1,1,Intervals$hdr[2])
  LowerInt<-Intervals$hdr[1]
  DFWorkingSubet$totalAvailDiscrete<-0
  DFWorkingSubet$totalAvailDiscrete[DFWorkingSubet$TotalAvail<LowerInt] <- 1
  DFWorkingSubet$totalAvailDiscrete[(DFWorkingSubet$TotalAvail>LowerInt)&(DFWorkingSubet$TotalAvail<=(LowerInt+(UpperInt-LowerInt)/NumIntervalDiscretise))] <- 2
  DFWorkingSubet$totalAvailDiscrete[(DFWorkingSubet$TotalAvail>(LowerInt+(UpperInt-LowerInt)/NumIntervalDiscretise))&(DFWorkingSubet$TotalAvail<=(LowerInt+(2*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 3
  DFWorkingSubet$totalAvailDiscrete[(DFWorkingSubet$TotalAvail>(LowerInt+(2*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalAvail<=(LowerInt+(3*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 4
  DFWorkingSubet$totalAvailDiscrete[(DFWorkingSubet$TotalAvail>(LowerInt+(3*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalAvail<=(LowerInt+(4*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 5
  DFWorkingSubet$totalAvailDiscrete[(DFWorkingSubet$TotalAvail>(LowerInt+(4*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalAvail<=(LowerInt+(5*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 6
  DFWorkingSubet$totalAvailDiscrete[(DFWorkingSubet$TotalAvail>(LowerInt+(5*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalAvail<=(UpperInt))]<-7
  print(min(DFWorkingSubet$totalAvailDiscrete))
  print(max(DFWorkingSubet$totalAvailDiscrete))
  print(UpperInt)
  print(LowerInt)
}

for (edge in AllEges) {
  print(edge)
  Intervals<-hdr(DFWorkingSubet$TotalCost11,prob=0.7)
  UpperInt<-Intervals$hdr[2]#ifelse(Intervals$hdr[2]>1,1,Intervals$hdr[2])
  LowerInt<-max(0,Intervals$hdr[1])
  print(UpperInt)
  print(LowerInt)
  DFWorkingSubet$TotalCost11Discrete<-0
  DFWorkingSubet$TotalCost11Discrete[DFWorkingSubet$TotalCost11<LowerInt] <- 1
  DFWorkingSubet$TotalCost11Discrete[(DFWorkingSubet$TotalCost11>LowerInt)&(DFWorkingSubet$TotalCost11<=(LowerInt+(UpperInt-LowerInt)/NumIntervalDiscretise))] <- 2
  DFWorkingSubet$TotalCost11Discrete[(DFWorkingSubet$TotalCost11>(LowerInt+(UpperInt-LowerInt)/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost11<=(LowerInt+(2*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 3
  DFWorkingSubet$TotalCost11Discrete[(DFWorkingSubet$TotalCost11>(LowerInt+(2*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost11<=(LowerInt+(3*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 4
  DFWorkingSubet$TotalCost11Discrete[(DFWorkingSubet$TotalCost11>(LowerInt+(3*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost11<=(LowerInt+(4*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 5
  DFWorkingSubet$TotalCost11Discrete[(DFWorkingSubet$TotalCost11>(LowerInt+(4*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost11<=(LowerInt+(5*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 6
  DFWorkingSubet$TotalCost11Discrete[(DFWorkingSubet$TotalCost11>(LowerInt+(5*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost11<=(UpperInt))]<-7
  print(min(DFWorkingSubet$TotalCost11Discrete))
  print(max(DFWorkingSubet$TotalCost11Discrete))
}
for (edge in AllEges) {
  print(edge)
  Intervals<-hdr(DFWorkingSubet$TotalCost11,prob=0.7)
  UpperInt<-Intervals$hdr[2]#ifelse(Intervals$hdr[2]>1,1,Intervals$hdr[2])
  LowerInt<-max(0,Intervals$hdr[1])
  print(UpperInt)
  print(LowerInt)
  DFWorkingSubet$TotalCost12Discrete<-0
  DFWorkingSubet$TotalCost12Discrete[DFWorkingSubet$TotalCost12<LowerInt] <- 1
  DFWorkingSubet$TotalCost12Discrete[(DFWorkingSubet$TotalCost12>LowerInt)&(DFWorkingSubet$TotalCost12<=(LowerInt+(UpperInt-LowerInt)/NumIntervalDiscretise))] <- 2
  DFWorkingSubet$TotalCost12Discrete[(DFWorkingSubet$TotalCost12>(LowerInt+(UpperInt-LowerInt)/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost12<=(LowerInt+(2*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 3
  DFWorkingSubet$TotalCost12Discrete[(DFWorkingSubet$TotalCost12>(LowerInt+(2*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost12<=(LowerInt+(3*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 4
  DFWorkingSubet$TotalCost12Discrete[(DFWorkingSubet$TotalCost12>(LowerInt+(3*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost12<=(LowerInt+(4*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 5
  DFWorkingSubet$TotalCost12Discrete[(DFWorkingSubet$TotalCost12>(LowerInt+(4*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost12<=(LowerInt+(5*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 6
  DFWorkingSubet$TotalCost12Discrete[(DFWorkingSubet$TotalCost12>(LowerInt+(5*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet$TotalCost12<=(UpperInt))]<-7
  print(min(DFWorkingSubet$TotalCost12Discrete))
  print(max(DFWorkingSubet$TotalCost12Discrete))
}

#common_col_names <- intersect(names(DFWorkingSubet11), names(DFWorkingSubet12))
#Merge<-merge(DFWorkingSubet11, DFWorkingSubet12, by=common_col_names, all.x=TRUE)

View(DFWorkingSubet)
















#####################################

jjj<-1
NewcolsVec=c("TotalAvailDisc","TotalCost11Disc","TotalCost12Disc","TotalCost21Disc","TotalCost22Disc")
DF[NewcolsVec]<-NA
View(DF)
for (i in colnames(DF)){
  if (i %in% list("TotalAvail","TotalCost11","TotalCost12","TotalCost21","TotalCost22")){
    for (j in colnames(DF)){
      if (j %in% list("TotalAvailDisc","TotalCost11Disc","TotalCost12Disc","TotalCost21Disc","TotalCost22Disc")){
        if (stri_detect_fixed(j, i)){
          
          #print(i)
          #Newcol<-print(paste(i,"Disc"))
          Newcol<-grep(j, colnames(DF))#paste(i)
          Target<-grep(i, colnames(DF))
          #DFWorkingSubet[ , ncol(DFWorkingSubet) + 1] <- 0# Append new column
          #colnames(DFWorkingSubet)[ncol(DFWorkingSubet)] <- Newcol
          for (edge in AllEges) {
            print(edge)
            DFWorkingSubet<-DF[DF$Edge == edge, ]  
            Intervals<-hdr(DFWorkingSubet[[Targetcol]],prob=0.7)
            print(Intervals)
            UpperInt<-ifelse(Intervals$hdr[2]>1,1,Intervals$hdr[2])
            LowerInt<-Intervals$hdr[1]
            #new <- rep(i, nrow(DFWorkingSubet)) 
            #DFWorkingSubet[ , ncol(DFWorkingSubet) + 1] <- new#0# Append new column
            #colnames(DFWorkingSubet)[ncol(DFWorkingSubet)] <- Newcol
            #DFWorkingSubet[,..Newcol]<-0
            DFWorkingSubet$j[DFWorkingSubet[i]<LowerInt] <- 1
            DFWorkingSubet$Newcol[(DFWorkingSubet[[Targetcol]]>LowerInt)&(DFWorkingSubet[[Targetcol]]<=(LowerInt+(UpperInt-LowerInt)/NumIntervalDiscretise))] <- 2
            DFWorkingSubet$Newcol[(DFWorkingSubet[[Targetcol]]>(LowerInt+(UpperInt-LowerInt)/NumIntervalDiscretise))&(DFWorkingSubet[[Targetcol]]<=(LowerInt+(2*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 3
            DFWorkingSubet$Newcol[(DFWorkingSubet[[Targetcol]]>(LowerInt+(2*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet[[Targetcol]]<=(LowerInt+(3*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 4
            DFWorkingSubet$Newcol[(DFWorkingSubet[[Targetcol]]>(LowerInt+(3*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet[[Targetcol]]<=(LowerInt+(4*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 5
            DFWorkingSubet$Newcol[(DFWorkingSubet[[Targetcol]]>(LowerInt+(4*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet[[Targetcol]]<=(LowerInt+(5*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 6
            DFWorkingSubet$Newcol[(DFWorkingSubet[[Targetcol]]>(LowerInt+(5*(UpperInt-LowerInt))/NumIntervalDiscretise))&(DFWorkingSubet[[Targetcol]]<=(UpperInt))]<-7
            #+(6*(UpperInt-LowerInt))/NumIntervalDiscretise))] <- 7
            #DFWorkingSubet$totalAvailDiscrete <- ifelse(DFWorkingSubet$TotalAvail<Intervals$hdr[1], 1, 10)
            print(min(DFWorkingSubet$Newcol))
            print(max(DFWorkingSubet$Newcol))
            print(UpperInt)
            print(LowerInt)
          }
        }
      }
    }
  }
}


View(DFWorkingSubet)

########
eet<-hdr(Alg2pd2OnlyAsset$TotalAvail,prob=0.9)[1][1]
typeof(eet)

min(DFWorkingSubet$totalAvailDiscrete)
View(DFWorkingSubet$TotalAvail[DFWorkingSubet$totalAvailDiscrete==0])
setwd("~/2019-05-30-NKJavad-NewRajab/Eindhoven-research/2021-01-01-FinalSelected/future-2/Case-data-02-FinalPythonCodeResults/20210820-Alg3Results/20211020ResultsAlg4")





syslevelcostaggre<-read.table("2021-10-21-SystemLevelCostsAggregatedAlg4.csv",header = TRUE,sep = ",")
fw11<-descdist(syslevelcostaggre$KemenyConst,boot = 1000)
fw <- fitdist(syslevelcostaggre$KemenyConst, "weibull")
summary(fw)
fitdist()

hdr(syslevelcostaggre$KemenyConst,prob=0.9)
hdr(syslevelcostaggre$KemenyConst,prob=0.8)
hdr(syslevelcostaggre$KemenyConst,prob=0.7)
hdr(syslevelcostaggre$KemenyConst,prob=0.6)
hdr(syslevelcostaggre$KemenyConst,prob=0.5)



hdr(syslevelcostaggre$KemenyConst,prob=0.9)
hdr(syslevelcostaggre$KemenyConst,prob=0.8)
hdr(syslevelcostaggre$KemenyConst,prob=0.7)
hdr(syslevelcostaggre$KemenyConst,prob=0.6)
hdr(syslevelcostaggre$KemenyConst,prob=0.5)


















