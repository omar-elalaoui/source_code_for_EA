library(ScottKnott)
library(MASS)
library(ggplot2)
library(readxl)
packageVersion("ScottKnott")
graphics.off()
par("mar") 


d1 <- read_excel("data_for_SK.xlsx", sheet = "mous")
d2 <- read_excel("data_for_SK.xlsx", sheet = "och")
d3 <- read_excel("data_for_SK.xlsx", sheet = "pho")

DT1<- aov(ACC ~ MODEL,data=d1)
DT2<- aov(ACC ~ MODEL,data=d2)
DT3<- aov(ACC ~ MODEL,data=d3)

sk1 <- SK(DT1,diseprsion="mm")
sk2 <- SK(DT2,diseprsion="mm")
sk3 <- SK(DT3,diseprsion="mm")

plot(sk1, xlab="", ylab="Accuracy", title='P. Moussieri',las = 2,cex=0.75)
plot(sk2, xlab="", ylab="Accuracy", title='P. Ochruros',las = 2,cex=0.75)
plot(sk3, xlab="", ylab="Accuracy", title='P. Phoenicurus',las = 2,cex=0.75)
