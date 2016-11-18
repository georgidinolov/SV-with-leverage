## input: load.save.input$data.file: file location of where the data
##                                   set has been saved
##        load.save.input$model.parameters.file: file location of
##                                               where the model
##                                               parameter object used
##                                               to generate he data
##                                               has been saved. This
##                                               contains the
##                                               model.parameter
##                                               object used as input
##                                               of the
##                                               generate.simulated.data(...)
##                                               function.
##        load.save.input$save.directory: directory where inputs and
##                                        output are saved. Name of
##                                        the file is indexed by
##                                        delta.t
##        random.walk: LOGICAL, if TRUE, phi is sampled with a random
##                     walk MH step. If FALSE, Laplace approximation
##                     proposal is used.
##        xi.2.override: LOGICAL, if TRUE, override the xi.square in
##                       the model.parameters object, and replace with
##                       xi.2.set.
##        xi.2.set: see xi.2.override
##        proposal.covariance: matrix (3x3) for the covariance of the
##                             proposal distribution for sampling phi
##                             = (alpha,theta,tau.square)
##        delta.t: sampling period
##        number.posterior.samples: int, number of MCMC samples taked
##                                  after burn-in
##        burn.in: number of MCMC samples to be discarded as burn in
##        number.paths.to.keep: number of posterior samples for to keep
##
## output: out$alpha.hats: posterior draws for alpha.hat    
##         out$theta.hats: posterior draws for theta.hat
##         out$tau.square.hats: posterior draws for tau.square.hat
##         out$log.volatilities.quantiles: array
##                                         (3 x number.posterior.samples). 1st
##                                         row is the approximate 2.5%
##                                         quantile for h, at each
##                                         samplig time. 2nd row is
##                                         the 50% quantile for each
##                                         time, and 3rd row is the
##                                         97.5% quantile.
##         out$h.true: true volatility path at sampling frequency.
##         out$IVs: Integrated volatilities
##         out$proposal.covariance.ll.mean: average of proposal
##                                          matrices used when random
##                                          walk is set to OFF
##
##         saves the inputs and output to a file in the save.directory
run.mcmc <- function(load.save.input,
                     random.walk = FALSE,
                     xi.2.override,
                     xi.2.set = 0,
                     proposal.covariance,
                     delta.t,
                     number.posterior.samples,
                     burn.in,
		     number.paths.to.keep) {
    library(mvtnorm);
    library(MCMCpack);
    source("/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/prior-elicitation.R");
    source("/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/sv-single-factor-no-rho-sampling-functions.R");

    ## number of elements in the Gaussian mixture representation of
    ## the 0.5log(chi^2) term
    M = 9;
    mixture.probabilities = c(0.00609,
                          0.04775,
                          0.13057,
                          0.20674,
                          0.22715,
                          0.18842,
                          0.12047,
                          0.05591,
                          0.01575,
                          0.00115);
    
    mixture.means = c(1.92677,
                      1.34744,
                      0.73504,
                      0.02266,
                      -0.85173,
                      -1.97278,
                      -3.46788,
                      -5.55246,
                      -8.68384,
                      -14.65000);
    
    mixture.variances = c(0.11265,
                          0.17788,
                          0.26768,
                          0.40611,
                          0.62699,
                          0.98583,
                          1.57469,
                          2.54498,
                          4.16591,
                          7.33342);
    
    load(load.save.input$data.file);
    load(load.save.input$model.parameters.file);

    ## GENERATE PRIORS START ##
    ##        alpha.prior.params: vector, first entry is the alpha prior
    ##                            mean and second alpha prior variance.
    ##        theta.prior.params: vector, first entry is the theta prior
    ##                            mean and second theta prior variance.
    ##        tau.square.prior.params: vector, first entry is the
    ##                                 tau.square prior shape, and second
    ##                                 entry is the prior rate parameter.
    alpha.hat.mean = model.parameters$alpha.hat;
    alpha.hat.var = (alpha.hat.mean * 10)^2;
    
    theta.hat.mean = model.parameters$theta.hat;
    theta.hat.var = (theta.hat.mean * 10)^2
    
    tau.square.hat.mean = model.parameters$tau.square.hat;
    tau.square.hat.var = (tau.square.hat.mean * 10)^2;
    
    source("/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/prior-elicitation.R");
    theta.prior.params <- theta.prior.parameters(theta.hat.mean,
                                                 theta.hat.var,
                                                 delta.t);
    tau.square.prior.params <- tau.square.prior.parameters(tau.square.hat.mean,
                                                           tau.square.hat.var,
                                                           theta.hat.mean,
                                                           theta.hat.var,
                                                           delta.t);
    alpha.prior.params <- c(alpha.hat.mean + 0.5*log(delta.t), alpha.hat.var);
    ## GENERATE PRIORS END ## 
    
    xi.2 = model.parameters$xi.square;
    if (xi.2.override == TRUE) {
        xi.2 = xi.2.set;
    }

    ## In case we are not using a random walk, keep track of the
    ## average proposal matrices used by the Laplace approximation
    ## method.
    proposal.covariance.ll.mean = diag(rep(0,3));
    
    delta.t.gen = model.parameters$delta.t.generation;
    nn = length(log.prices.and.log.volatilities$log.prices);
    
    log.prices = log.prices.and.log.volatilities$
    log.prices[seq(1,nn,by = delta.t/delta.t.gen)];
    
    log.latent.prices = log.prices.and.log.volatilities$
    log.prices[seq(1,nn,by = delta.t/delta.t.gen)];

    n = length(log.prices);
    
    y.star = 0.5* log((log.latent.prices[seq(2,n)] -
                       log.latent.prices[seq(1,n-1)])^2);

    h.true = log.prices.and.log.volatilities$
    log.sigma[seq(1,
                  nn,
                  by = delta.t/delta.t.gen)] -
        0.5*log(delta.t.gen) +
        0.5*log(delta.t);

    h = h.true;
    
    gammas = rep(0,n-1);
    tau.square.hat = model.parameters$tau.square.hat;
    theta.hat = model.parameters$theta.hat;
    
    ## DISCRETE-TIME PARAMS ##
    theta = exp(-model.parameters$theta.hat*delta.t);
    tau.square = model.parameters$tau.square.hat /
        (2*model.parameters$theta.hat) *
        (1 - exp(-2*model.parameters$theta.hat*delta.t));
    alpha = model.parameters$alpha.hat + 0.5*log(delta.t);

    ## TAU SQUARE TRUE START ##
    tau.square.true = tau.square;
    ## TAU SQUARE TRUE END ##

    ## latent signal sums and squared sums, used for estimating
    ## posterior mean and std. dev. at each observational time.
    log.filtered.prices.sums = rep(0, n);
    log.filtered.prices.square.sums = rep(0, n);
    
    log.volatilities.sums = rep(0, n);
    log.volatilities.square.sums = rep(0, n);

    ## posterior draws
    alpha.hats <- rep(0, number.posterior.samples);
    theta.hats <- rep(0, number.posterior.samples);
    tau.square.hats <- rep(0, number.posterior.samples);
    IVs <- rep(0, number.posterior.samples);
    RVs <- rep(0, number.posterior.samples);
    
    if (number.paths.to.keep < number.posterior.samples) {
       path.samples <- vector(mode="list", length=number.paths.to.keep);
    } else {
      path.samples <- vector(mode="list", length=number.posterior.samples);
    }
    
    for (i in seq(1,number.posterior.samples + burn.in)) {
        ## SAMPLING BLOCK 1 START ##
        log.latent.prices <- sample.latent.prices(log.prices,
                                                  xi.2,
                                                  h);
        y.star <- 0.5*log((log.latent.prices[seq(2,n)] - log.latent.prices[seq(1,n-1)])^2);

        gammas <- sample.gammas(y.star,
                                h,
                                mixture.means,
                                mixture.variances);
        ## SAMPLING BLOCK 1 END ##

        ## SAMPLING BLOCK 2 END ##
        if (random.walk) {
            phi <- sample.phi.rw(alpha,
                                 theta,
                                 tau.square,
                                 h,
                                 y.star,
                                 gammas,
                                 mixture.means,
                                 mixture.variances,
                                 alpha.prior.params,
                                 theta.prior.params,
                                 tau.square.prior.params,
                                 proposal.covariance,
                                 delta.t);
        } else {
            phi.output <- sample.phi(alpha,
                                     theta,
                                     tau.square,
                                     h,
                                     y.star,
                                     gammas,
                                     mixture.means,
                                     mixture.variances,
                                     alpha.prior.params,
                                     theta.prior.params,
                                     tau.square.prior.params,
                                     proposal.covariance,
                                     delta.t);
            phi <- phi.output$phi;
            if (i > burn.in) {
                proposal.covariance.ll.mean =
                    (proposal.covariance.ll.mean * ((i-1)-burn.in) +
                     phi.output$proposal.covariance.ll) / (i-burn.in)
            }
        }

        alpha <- phi[1];
        theta <- phi[2];
        tau.square <- phi[3];
        
        h <- sample.h(alpha,
                      theta,
                      tau.square,
                      y.star,
                      gammas,
                      mixture.means,
                      mixture.variances);
        ## SAMPLING BLOCK 2 END ##
        
        print(i);

        alpha.hat = alpha - 0.5*log(delta.t);
        theta.hat = log(theta)/(-delta.t);
        tau.square.hat = tau.square / (1-exp(-2*theta.hat*delta.t)) * (2*theta.hat);

        ## record posterior draws after burn-in ##
        if (i > burn.in) {
            alpha.hats[i-burn.in] = alpha.hat;
            theta.hats[i-burn.in] = theta.hat;
            tau.square.hats[i-burn.in] = tau.square.hat;
            IVs[i-burn.in] = sum(exp(2*h));
            RVs[i-burn.in] = sum((log.latent.prices[-1]-
			log.latent.prices[-length(log.latent.prices)])^2);

	    if ( sum(round(seq(1,
			number.posterior.samples,
			length.out=number.paths.to.keep))  
		== (i-burn.in)) > 0 ) {
	    	
		path.samples[[i-burn.in]] = h;
	    }

            for (j in seq(1,n)) {
                log.filtered.prices.sums[j] = log.filtered.prices.sums[j] +
                    log.latent.prices[j];
                log.filtered.prices.square.sums[j] = log.filtered.prices.square.sums[j] +
                    log.latent.prices[j]^2;

                log.volatilities.sums[j] = log.volatilities.sums[j] +
                    h[j];
                log.volatilities.square.sums[j] = log.volatilities.square.sums[j] +
                    h[j]^2;
            }
        }
        
        print(c(alpha.hat, theta.hat, tau.square.hat));
    }

    ## Mean and approximate 0.025 and 0.975 (mean +/- 1.96*SD)
    ## quantiles will be stored here.
    log.filtered.prices.quantiles = array(dim=c(3,n));
    log.volatilities.quantiles = array(dim=c(3,n));

    for (i in seq(1,n)) {
        log.filtered.prices.mean = log.filtered.prices.sums[i]/number.posterior.samples;
        log.filtered.prices.var = log.filtered.prices.square.sums[i]/(number.posterior.samples-1) -
            log.filtered.prices.sums[i]^2/(number.posterior.samples*(number.posterior.samples-1));
        
        log.filtered.prices.quantiles[1,i] = log.filtered.prices.mean -
            1.96*sqrt(log.filtered.prices.var);
        log.filtered.prices.quantiles[2,i] = log.filtered.prices.mean;
        log.filtered.prices.quantiles[3,i] = log.filtered.prices.mean +
            1.96*sqrt(log.filtered.prices.var);

        log.volatilities.mean = log.volatilities.sums[i]/number.posterior.samples;
        log.volatilities.var = log.volatilities.square.sums[i]/(number.posterior.samples-1) -
            number.posterior.samples/(number.posterior.samples-1)*log.volatilities.mean^2;
        
        log.volatilities.quantiles[1,i] = log.volatilities.mean -
            1.96*sqrt(log.volatilities.var);
        log.volatilities.quantiles[2,i] = log.volatilities.mean;
        log.volatilities.quantiles[3,i] = log.volatilities.mean +
            1.96*sqrt(log.volatilities.var);
    }

    ## Plots for log-vol path and scatterplots for (alpha, theta,
    ## tau^2).
    plot(log.volatilities.quantiles[2,],
         ylim = c(min(min(log.volatilities.quantiles[1,]), min(h.true)),
                  max(max(log.volatilities.quantiles[3,]), max(h.true))),
         type = "l");
    coord.x = c(seq(1,n), seq(n,1));
    coord.y = c(log.volatilities.quantiles[3,seq(1,n)],
                log.volatilities.quantiles[1,seq(n,1)]);
    polygon(coord.x, coord.y, col = "grey");
    lines(log.volatilities.quantiles[2,], lwd = 2);
    lines(h.true, col = "red", lwd = 2);

    print(xi.2);
    out = NULL;
    out$alpha.hats = alpha.hats;
    out$theta.hats = theta.hats;
    out$tau.square.hats = tau.square.hats;
    out$log.volatilities.quantiles = log.volatilities.quantiles;
    out$h.true = h.true;
    out$IVs = IVs;
    out$RVs = RVs;
    out$proposal.covariance.ll.mean = proposal.covariance.ll.mean;
    out$path.samples = path.samples;

    IV.true = sum(exp(2*log.prices.and.log.volatilities$log.sigma));

    save(file = paste(load.save.input$save.directory,
                      "posterior-results-deltat-",
                      delta.t, ".Rdata", sep = ""),
         list = c("load.save.input",
                  "random.walk",
                  "xi.2.override",
                  "xi.2.set",
                  "alpha.prior.params",
                  "theta.prior.params",
                  "tau.square.prior.params",
                  "proposal.covariance",
                  "delta.t",
                  "number.posterior.samples",
                  "burn.in",
                  "IV.true",
                  "out"));

    pdf(file = paste(load.save.input$save.directory, "scatterplots-",
                     delta.t, ".pdf", sep = ""))
    par(mfrow=c(3,3));
    traceplot(as.mcmc(alpha.hats));
    abline(h = model.parameters$alpha.hat, col = "red");
    plot(alpha.hats, theta.hats);
    plot(alpha.hats, tau.square.hats);
    
    plot.new();
    traceplot(as.mcmc(theta.hats));
    abline(h = model.parameters$theta.hat, col = "red");
    plot(theta.hats, tau.square.hats);
    
    plot.new();
    plot.new();
    traceplot(as.mcmc(tau.square.hats));
    abline(h = model.parameters$tau.square.hat, col = "red");
    
    dev.off();

    pdf(file = paste(load.save.input$save.directory, "log-vol-path-",
                     delta.t, ".pdf", sep = ""))
    plot(log.volatilities.quantiles[2,],
         ylim = c(min(min(log.volatilities.quantiles[1,]), min(h.true)),
                  max(max(log.volatilities.quantiles[3,]), max(h.true))),
         type = "l");
    coord.x = c(seq(1,n), seq(n,1));
    coord.y = c(log.volatilities.quantiles[3,seq(1,n)],
                log.volatilities.quantiles[1,seq(n,1)]);
    polygon(coord.x, coord.y, col = "grey");
    lines(log.volatilities.quantiles[2,], lwd = 2);
    lines(h.true, col = "red", lwd = 2);

    for (i in seq(1,length(path.samples))) {
    	lines(path.samples[[i]], col = "blue");
    }

    dev.off();
    
    return(out);
}
