## input: model.parameters (object)
##        model.parameters$alpha.hat: mean level of the
##                                    continuous-time log-volatility
##                                    process
##        model.parameters$tau.square.hat: variance of the
##                                         continuous-time
##                                         log-volatility process
##        model.parameters$theta.hat: inverse mean-reversion timescale
##                                    of the continuous-time
##                                    log-volatility process
##        model.parameters$xi.square: observational noise variance
##        model.parameters$TT: observational duration
##        model.parameters$delta.t.generation: time discretization of
##                                             the continuous model
##                                             for the purpose of data
##                                             generation
##        save.path: directory (as string) where the simulated data and model
##                   parameters are to be saved
## output: NULL, no vals returned. However, there are 3 .Rdata files
##         saved in the save.path directory:
##         model-parameters-*.Rdata: contains the model.parameters
##                                   object used to generate the
##                                   simulated data.
##         simulated-prices-and-returns-no-noise-*.Rdata: contains the
##                  object log.prices.and.log.volatilities, with
##                  elements:
##                  log.prices.and.log.volatilities$log.sigma:
##                     timeseries of the h_t vector. To change to the
##                     continuous-time scale, subtract
##                     0.5*log(\Delta_{gen}) from each element of the
##                     vector
##                  log.prices.and.log.volatilities$log.prices:
##                     timeseries of the true prices.
##         simulated-prices-and-returns-added-noise-*.Rdata: contains the
##                  object log.prices.and.log.volatilities, with
##                  contents:
##                  log.prices.and.log.volatilities$log.sigma:
##                     timeseries of the h_t vector. To change to the
##                     continuous-time scale, subtrant
##                     0.5*log(\Delta_{gen}) from each element.
##                  log.prices.and.log.volatilities$log.prices:
##                     timeseries of the NOISE CONTAMINATED prices.
##         The * in the above files stands for a timestamp of when the
##         data was generated
generate.simulated.data <- function(model.parameters, save.path){
  time <- Sys.time();
  now <- unclass(as.POSIXlt(time));
  ## date (string) is a timestamp of when the data is generated. Unless the
  ## function is run in parallel, this ensures the multiple data sets
  ## are distinguished.
  date = paste(now$mon+1, "-", now$mday, "-", now$hour, "-",
      now$min, "-", round(now$sec), sep="");
  fileName = paste("simulated-prices-and-returns-added-noise-", date, ".Rdata", sep = "");

  ## saving the model generating parameters (as the model.parameters
  ## object) with the date timestamp in the save.path directory.
  save(file = paste(save.path, "model-parameters-", date, ".Rdata", sep = ""),
       list = c("model.parameters"));
  
  ## Generating data with a delta.t period
  TT = model.parameters$TT;
  delta.t = model.parameters$delta.t.generation;
  log.prices.and.log.volatilities = NULL;
  log.prices.and.log.volatilities$times = seq(1,floor(TT/delta.t));
  n = length(log.prices.and.log.volatilities$times);

  theta.hat = model.parameters$theta.hat;
  tau.square.hat = model.parameters$tau.square.hat;
  alpha.hat = model.parameters$alpha.hat;
  
  ## discrete-time parameters ##
  theta = (1-theta.hat*delta.t);
  tau.square = tau.square.hat * delta.t;
  alpha = alpha.hat + 0.5*log(delta.t);
  xi.2 = model.parameters$xi.square;

  epsilon1s = rnorm(n=n+1, mean = 0, sd = 1);
  epsilon2s = rnorm(n=n+1, mean = 0, sd = 1);
    
  log.prices = rep(0,(n+1));
  log.sigma = rep(NA,(n+1));

  ## the initial prices level is $100. The initial (discrete)
  ## log-volatility level is the average (discrete) log-volatility.
  log.price.t.minus.1 = log(100);
  log.sigma.t = alpha;
    
  log.prices[1] = log.price.t.minus.1;
  log.sigma[1] = log.sigma.t;
  
  for( i in seq(2,(n+1)) ){
      log.price.t = log.price.t.minus.1 + exp(log.sigma.t) * epsilon1s[i];
      
      if (abs(log.price.t) == Inf) {
          break;
      }
      
      log.sigma.t.plus.1 = alpha + theta*(log.sigma.t - alpha) +
          sqrt(tau.square)*epsilon2s[i];
      
      log.price.t.minus.1 = log.price.t;
      log.sigma.t = log.sigma.t.plus.1;
      
      log.prices[i] = log.price.t;
      log.sigma[i] = log.sigma.t.plus.1;
      if(i %% 1000 == 0) {
          print(i);
      }
  }
  
  log.prices.and.log.volatilities$log.sigma = log.sigma;
  log.prices.and.log.volatilities$log.prices = log.prices;
  log.prices.true = log.prices;
  
  ##### saving data with no noise #####
  fileName = paste("simulated-prices-and-returns-no-noise-", date, ".Rdata", sep = "");
  
  save(file = paste(save.path, fileName, sep = ""),
       list = c("log.prices.and.log.volatilities"));
  
  #### Adding the noise according to the model #### 
  fileName = paste("simulated-prices-and-returns-added-noise-", date, ".Rdata", sep = "")
  errors = rnorm(n=n+1,mean = 0, sd = sqrt(xi.2));
  prices.plus.errors = log.prices + errors;
  log.prices.and.log.volatilities$log.prices = prices.plus.errors;

  save(file = paste(save.path, fileName, sep = ""),
       list = c("log.prices.and.log.volatilities"));
  ## ##
  
  ## plotting the true and noisy prices every second ##
  noisy.prices = prices.plus.errors[seq(1,n+1, by = max(1,1000/delta.t))];
  true.prices = log.prices.true[seq(1,n+1, by = max(1,1000/delta.t))];
  
  pdf(paste(save.path, "log-prices-noisy-true-", date,".pdf", sep = ""), 6, 6);
  plot(seq(1,length(noisy.prices))/(60*60), noisy.prices, type = "l",
       xlab = "time (hours)",
       ylab = "log prices");
  lines(seq(1,length(noisy.prices))/(60*60), true.prices, col = "red",
        lty = "dashed");
  dev.off();
  ## ##

  ## plotting the volatility path every second, on the continuous (hat) scale ##
  log.sigma.second = log.sigma[seq(1,n+1, by = max(1,1000/delta.t))];
  log.sigma.hat = log.sigma.second - 0.5*log(delta.t);
  
  pdf(paste(save.path, "log-volatilities-", date,".pdf", sep = ""), 6, 6);
  plot(seq(1,length(log.sigma.hat))/(60*60), log.sigma.hat, type = "l",
       xlab = "time (hours)",
       ylab = "log(sigma_hat)");
  dev.off();
  ## ##
}


## input: model.parameters (object)
##        model.parameters$alpha.hat: mean level of the
##                                    continuous-time log-volatility
##                                    process
##        model.parameters$tau.square.hat: variance of the
##                                         continuous-time
##                                         log-volatility process
##        model.parameters$theta.hat: inverse mean-reversion timescale
##                                    of the continuous-time
##                                    log-volatility process
##        model.parameters$xi.square: observational noise variance
##        model.parameters$TT: observational duration
##        model.parameters$delta.t.generation: time discretization of
##                                             the continuous model
##                                             for the purpose of data
##                                             generation
##        save.path.list: List of directories (as strings) where the
##                        simulated data sets and model parameters are
##                        to be saved.
##        number.threads: how many threads are to be called
generate.multiple.simulated.data <- function(model.parameters,
                                             save.path.list,
                                             number.clusters) {
    library(parallel);
    
    ## checking if each save path in the list exists; if it doesn't,
    ## create it.
    for (save.path in save.path.list) {
    	print(save.path);
        if (file.exists(save.path) == FALSE) {
            print(paste(save.path, " doesn't exist! Creating one", sep = ""));
            dir.create(save.path, showWarnings = FALSE);
        }
    }

    cl <- makeCluster(number.clusters);
    parLapply(cl = cl,
              X = save.path.list,
              fun = generate.simulated.data,
              model.parameters = model.parameters);
    stopCluster(cl);
}
