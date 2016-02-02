# Geo rank

library(data.table)
x1 <- fread("./output/ens_bag5_20160131_seed132.csv", data.table = F)
x2 <- fread("./output/sub46.csv", data.table = F)

x1[,2] <- rank(x1[,2])/nrow(x1)
x2[,2] <- rank(x2[,2])/nrow(x2)
# check that the index ordering matches :-)
xfor <- x1
xfor[,2] <- exp(0.5 * log(x1[,2]) + 0.5 * log(x2[,2]))
# save, submit, (hopefully) smile
write.csv(xfor, "./submissions/xmix_01022016_5050.csv", row.names = F)
