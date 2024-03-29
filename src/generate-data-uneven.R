extract.data <- function(stock.of.interest, date, load.directory, save.directory){
  ## library(data.table)
  #stock.of.interest = "IBM             " #; 16 character string
  #date = [year][month][day], ex: 20090707; date = 20090707
  #load.directory = "C:/Research/SV-with-leverage/high-frequency-model-12-18-2014/"
  #save.directory = "C:/Classes/"
  
  day =  paste("taqtrade", date, sep = "")

  ##### run only if file exists in the directory #### 
  if( file.exists(paste(load.directory, day, sep = "")) == 1 ){
   
    first.index.approx <- function(current.index, mmin, mmax){
      # a recursive function for finding the first index of the target stock
      
      print(c(current.index, mmax-mmin))
      data.line = fread(input = paste(load.directory, day, sep = ""), sep = "\n", header = F, nrows = 1, skip = current.index+1)
      #data.line = data[current.index,]
      current.symbol = substr(data.line[1,], 11, 26)
      print(c(current.symbol, stock.of.interest))
      
      oo = order(c(current.symbol, stock.of.interest))
      print(oo)
      
      if(mmax-mmin <= 5e6+1){
        
        #out = data[seq(mmin,mmax),]
        #out = c(mmin,mmax,current.index)
        out = mmin
         
      }else if(current.symbol == stock.of.interest){ # if the symbol is the right one, keep going up
        mmin = mmin
        mmax = current.index
        #current.index = round( ( mmax + mmin )/2)
        current.index = ceiling( ( mmax + mmin )/2)
        
        out = first.index.approx(current.index, mmin, mmax)
      }else{
        
        if(oo[1] == 2){ # the current symbol is greater than the target symbol
          
          mmin = mmin
          mmax = current.index
          current.index = round( ( mmax + mmin )/2 )
          
          out = first.index.approx(current.index, mmin, mmax)
          
        }else if(oo[1] == 1){ # the current symbol is less than the target symbol
          mmin = current.index
          mmax = mmax
          current.index = round( ( mmax + mmin )/2 )
          
          out = first.index.approx(current.index, mmin, mmax)
        }  
      }
      
      return(out)
    }
    
    first.index <- function(current.index, mmin, mmax, data ){
      # a recursive function for finding the first index of the target stock
      
      print(c(current.index, mmax-mmin))
      #data = fread(input = paste(load.directory, day, sep = ""), sep = "\n", header = F, nrows = 1, skip = current.index+1)
      data.line = data[current.index,]
      current.symbol = substr(data.line[1,], 11, 26)
      print(c(current.symbol, stock.of.interest))
      
      oo = order(c(current.symbol, stock.of.interest))
      print(oo)
      
      if(mmax-mmin <= 1){
        
        #out = data[seq(mmin,mmax),]
        #out = c(mmin,mmax,current.index)
        out = current.index
        
      }else if(current.symbol == stock.of.interest){ # if the symbol is the right one, keep going up
        mmin = mmin
        mmax = current.index
        #current.index = round( ( mmax + mmin )/2)
        current.index = ceiling( ( mmax + mmin )/2)
        
        out = first.index(current.index, mmin, mmax, data)
      }else{
        
        if(current.symbol == stock.of.interest){ # if the symbol is the right one, keep going DOWN
          mmin = current.index
          mmax = mmax
          #current.index = round( ( mmax + mmin )/2)
          current.index = floor( ( mmax + mmin )/2)
          
          out = last.index(current.index, mmin, mmax, data)
        }else if(oo[1] == 2){ # the current symbol is greater than the target symbol
          
          mmin = mmin
          mmax = current.index
          current.index = round( ( mmax + mmin )/2 )
          
          out = first.index(current.index, mmin, mmax, data)
          
        }else if(oo[1] == 1){ # the current symbol is less than the target symbol
          mmin = current.index
          mmax = mmax
          current.index = round( ( mmax + mmin )/2 )
          
          out = first.index(current.index, mmin, mmax, data)
        }  
      }
      
      return(out)
    }
    
    last.index.approx <- function(current.index, mmin, mmax ){
      # a recursive function for finding the first index of the target stock
      
      print(c(mmin, current.index, mmax, mmax-mmin))
      data.line = fread(input = paste(load.directory, day, sep = ""), sep = "\n", header = F, nrows = 1, skip = current.index+1)
      #data.line = data[current.index,]
      current.symbol = substr(data.line[1,], 11, 26)
      print(c(current.symbol, stock.of.interest))
      
      oo = order(c(current.symbol, stock.of.interest))
      print(oo)
      
      if(mmax-mmin <= 5e6+1){
        
        #out = data[seq(mmin,mmax),]
        #out = c(mmin,mmax,current.index)
        out = mmax
        
      }else if(current.symbol == stock.of.interest){ # if the symbol is the right one, keep going DOWN
        mmin = current.index
        mmax = mmax
        #current.index = round( ( mmax + mmin )/2)
        current.index = floor( ( mmax + mmin )/2)
        
        out = last.index.approx(current.index, mmin, mmax)
      }else{
        
        if(oo[1] == 2){ # the current symbol is greater than the target symbol
          
          mmin = mmin
          mmax = current.index
          current.index = round( ( mmax + mmin )/2 )
          
          out = last.index.approx(current.index, mmin, mmax)
          
        }else if(oo[1] == 1){ # the current symbol is less than the target symbol
          mmin = current.index
          mmax = mmax
          current.index = round( ( mmax + mmin )/2 )
          
          out = last.index.approx(current.index, mmin, mmax)
        }  
      }
      
      return(out)
    }
    
    last.index <- function(current.index, mmin, mmax, data ){
      # a recursive function for finding the first index of the target stock
      
      print(c(mmin, current.index, mmax, mmax-mmin))
      #data = fread(input = paste(load.directory, day, sep = ""), sep = "\n", header = F, nrows = 1, skip = current.index+1)
      data.line = data[current.index,]
      current.symbol = substr(data.line[1,], 11, 26)
      print(c(current.symbol, stock.of.interest))
      
      oo = order(c(current.symbol, stock.of.interest))
      print(oo)
      
      if(mmax-mmin <= 1){
        
        #out = data[seq(mmin,mmax),]
        #out = c(mmin,mmax,current.index)
        out = current.index
        
      }else if(current.symbol == stock.of.interest){ # if the symbol is the right one, keep going DOWN
        mmin = current.index
        mmax = mmax
        #current.index = round( ( mmax + mmin )/2)
        current.index = floor( ( mmax + mmin )/2)
        
        out = last.index(current.index, mmin, mmax, data)
      }else{
        
        if(oo[1] == 2){ # the current symbol is greater than the target symbol
          
          mmin = mmin
          mmax = current.index
          current.index = round( ( mmax + mmin )/2 )
          
          out = last.index(current.index, mmin, mmax, data)
          
        }else if(oo[1] == 1){ # the current symbol is less than the target symbol
          mmin = current.index
          mmax = mmax
          current.index = round( ( mmax + mmin )/2 )
          
          out = last.index(current.index, mmin, mmax, data)
        }  
      }
      
      return(out)
    }
    
    mmin = 1
    mmax = 40e6
    current.index = round( ( mmax + mmin )/2 + 1)
    
    fi.approx <- first.index.approx(current.index, mmin, mmax)
    li.approx <- last.index.approx(current.index, mmin, mmax)
    
    mmin = 1
    mmax = li.approx - fi.approx
    current.index = round( ( mmax + mmin )/2 + 1)
    data = fread(input = paste(load.directory, day, sep = ""), sep = "\n", header = F, nrows = li.approx - fi.approx, skip = fi.approx + 1)
    
    fi = first.index(current.index, mmin, mmax, data)
    li = last.index(current.index, mmin, mmax, data)
    
    data = data[seq(fi,li),]
    data = data$V1
    #data = read.table(file = paste(load.directory, day, sep = ""), header = T, skip = fi-1+fi.app, nrows = li-fi+1, sep = ",")
    
    LL = li-fi+1
    
    ########## taking out the derivative priced 
    sale.conditions = substr( data[seq(1,LL)], 27, 30)
    
    compare.symbols <- function(l){
      
      if( sum(l == "4") > 0 | sum(l == "V") > 0 | sum(l == "O") > 0 ){
        return(T)
      }else{
        return(F)}
      
    }
    
    
    ind = which( unlist( lapply( strsplit( sale.conditions, "" ), compare.symbols ) ) == T )
    
    data = data[-ind]
    
    LL = length(data)
    sale.conditions = substr( data[seq(1,LL)], 27, 30)
    ##################
    
    
    times.unformatted = substr( data[seq(1,LL)], 1, 9)
    
    prices.unformatted = substr( data[seq(1,LL)], 40, 50)
    
    times.milliseconds = (as.integer(substr(times.unformatted, 1,2)) - 9.5)*60*60*1000 + 
      (as.integer(substr(times.unformatted, 3,4)) - 0)*60*1000 + 
      (as.integer(substr(times.unformatted, 5,6)) - 0)*1000 +   
      (as.integer(substr(times.unformatted, 7,9)) - 0)
    
    ### ordering the times and prices
    sorting = sort(x = times.milliseconds, decreasing = F, index.return = T)
    
    prices.unformatted = prices.unformatted[sorting$ix]
    times.milliseconds = times.milliseconds[sorting$ix]
    ### 
    
    in.trading.day = which(times.milliseconds < (16-9.5)*60*60*1000 & times.milliseconds > 0)
    
    
    
    log.prices = log( as.integer(substr(prices.unformatted,1,7)) + 
                        1e-4*as.integer(substr(prices.unformatted,8,11)) )
    
    ind = which(times.milliseconds < 0)
    ll = ind[length(ind)]
    
    opening.price = log.prices[ll]
    
    log.prices = log.prices[in.trading.day]
    times.milliseconds = times.milliseconds[in.trading.day]
    
    ### log prices
    #par(mfrow = c(2,1))
    #plot(times.milliseconds/(1), log.prices, type = "l", main= "log prices", xlab = "milliseconds into trading day")
    
    
    #### log returns
    #log.returns = log.prices[-1] - log.prices[-length(log.prices)]
    #plot(times.milliseconds[-1]/(1), log.returns, type = "l", xlab = "milliseconds into trading day", main = "log returns")

    ## TODO(georgid): We still need the log returns!!!
    log.returns.milliseconds = log.prices.millisecond.grid[-1] - log.prices.millisecond.grid[-length(log.prices.millisecond.grid)]
    
    ### now we have a return for milliseconds t = 1, 2, 3, and NOT for t = 2, 3, ... b/c we included the opening price
    #################
    
    #out = NULL
    #out$log.prices.millisecond.grid = log.prices.millisecond.grid
    #out$milliseconds = milliseconds
    #out$log.returns.milliseconds = log.returns.milliseconds
    #return(out)
    
    prices.and.returns = list( times.milliseconds =
        times.milliseconds, log.prices = log.prices)
    
    save(list = c("prices.and.returns"), file = paste(save.directory, date, "-prices-and-returns.Rdata", sep = ""))
  }else{
    print(paste( paste(load.directory, day, sep = "") , " does not exist", sep = ""))
  }

}

stock.of.interest = "SPY"

filler = ""
for( ii in seq(1, 16-nchar(stock.of.interest))){
  filler = paste(filler, " ", sep = "")
}

for( date in c( seq(20100506,20100506) )  ){
 
  load.directory = paste("/share/Arbeit/gdinolov/real-data/EQY_US_ALL_TRADE_2010/EQY_US_ALL_TRADE_2010",  substr(date,5,6), "/", sep="")
  save.directory = paste( "/share/Arbeit/gdinolov/real-data/", stock.of.interest, "/", stock.of.interest, "2010/", stock.of.interest,
                    "2010", substr(date,5,6), "/", sep = "")
  
  extract.data( paste( stock.of.interest, filler, sep = "" ), date, load.directory, save.directory)
  
}


#data <- extract.data(stock.of.interest, date)
#
#log.returns.milliseconds  <- data$log.returns.milliseconds
#milliseconds              <- data$milliseconds
#log.prices.milliseconds   <- data$log.prices.millisecond.grid
#
#length(milliseconds)
#length(log.returns.milliseconds)
#length(log.prices.milliseconds)
#
#
#
###### 1-second scale
#TT = length(milliseconds)
#
#one.second.times          <- milliseconds[seq(1*1000, TT, by = 1*1000)]/(1000)
#log.prices.one.second     <- log.prices.milliseconds[seq(1, length(log.prices.milliseconds), by = 1000)]
#log.returns.one.second    <- log.prices.one.second[-1] - log.prices.one.second[-length(log.prices.one.second)]
#
#length(one.second.times)
#length(log.returns.one.second)
#length(log.prices.one.second)
#
#
#par(mfrow = c(2,1))
#plot(c(0,one.second.times), log.prices.one.second, type = "l", main = "log prices per second", ylab = "", xlab = "seconds")
#plot(one.second.times, log.returns.one.second, type = "l", main = "log returns per second", ylab = "", xlab = "seconds")
#
###### 5-second scale
#TT = length(milliseconds)
#
#five.second.times        <- milliseconds[seq(5*1000, TT, by = 5*1000)]/(1000)
#log.prices.five.second   <- log.prices.milliseconds[seq(1, length(log.prices.milliseconds), by = 5*1000)]
#log.returns.five.second  <- log.prices.five.second[-1] - log.prices.five.second[-length(log.prices.five.second)]
#
#length(five.second.times)
#length(log.returns.five.second)
#length(log.prices.five.second)
#
#par(mfrow = c(2,1))
#plot(c(0,five.second.times), log.prices.five.second, type = "l", main = "log prices every 5 seconds", ylab = "", xlab = "seconds")
#plot(five.second.times, log.returns.five.second, type = "l", main = "log returns every 5 seconds", ylab = "", xlab = "seconds")
#
##### 1 minute scale
#TT = length(milliseconds)
#
#one.minute.times         <- milliseconds[seq(60*1000, TT, by = 60*1000)]/(60*1000)
#log.prices.one.minute   <- log.prices.milliseconds[seq(1, length(log.prices.milliseconds), by = 60*1000)]
#log.returns.one.minute  <- log.prices.one.minute[-1] - log.prices.one.minute[-length(log.prices.one.minute)]
#
#length(one.minute.times)
#length(log.returns.one.minute)
#length(log.prices.one.minute)
#
#par(mfrow = c(2,1))
#plot(c(0,one.minute.times), log.prices.one.minute, type = "l", main = "log prices every minute", ylab = "", xlab = "minutes")
#plot(one.minute.times, log.returns.one.minute, type = "l", main = "log returns every minute", ylab = "", xlab = "minutes")
#
#
#
#prices.and.returns = list( milliseconds = milliseconds,
#                           log.returns.milliseconds = log.returns.milliseconds,
#                           log.prices.milliseconds  = log.prices.milliseconds)
#

