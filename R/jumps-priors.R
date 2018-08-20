rm(list=ls())
##
require("data.table")
spy <- fread("../data/GSPC.csv")
spy[, Date := as.Date(Date, "%Y-%m-%d")]
spy[, day_of_week := weekdays(Date)]
##
##
friday.close <- spy[day_of_week == "Friday",
                    .(Close, Date, next.Monday=(Date+3))]
monday.open <- spy[day_of_week == "Monday", .(Open, Date)]
##
##
setkey(friday.close, next.Monday)
setkey(monday.open, Date)
##
##
weekends <- friday.close[monday.open, nomatch=0]
##
weekends[, log.jumps := log(Close)-log(Open)]
