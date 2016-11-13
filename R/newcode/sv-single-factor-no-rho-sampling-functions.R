## Follows section 2.2 in the document.
sample.gammas <- function(y.star,
                          h,
                          mixture.means,
                          mixture.variances) {
    n = length(h);
    M = length(mixture.means);
    gammas = rep(0,n-1);
    log.probabilities = rep(0,M);
        for (i in seq(1,n-1)) {
            for (j in seq(1,M)) {
                log.probabilities[j] = dnorm(x = y.star[i] - h[i],
                                             mean = mixture.means[j]/2,
                                             sd = sqrt(mixture.variances[j]/4),
                                             log = TRUE);
            }
            ## print(log.probabilities);
            probability.vector = exp(log.probabilities - max(log.probabilities));
            gammas[i] = sample(x = seq(1,M), size = 1,
                               prob = probability.vector);
        }
    return(gammas);
}

## Follows section 2.4 in document.
sample.h <- function(alpha,
                     theta,
                     tau.square,
                     y.star,
                     gammas,
                     mixture.means,
                     mixture.variances) {
    
    n = length(y.star) + 1;
    h = rep(NA, n);
    
    mu.current = alpha;
    sigma2.current = tau.square / (1-theta);
    
    mus = rep(NA, n-1);
    sigma2s = rep(NA, n-1);
    
    ## FORWARD FILTER ##
    for (i in seq(1,n-1)) {
        m.current = alpha*(1-theta) + theta*mu.current;
        s2.current = tau.square + theta^2*sigma2.current;
        
        mu.current =
            y.star[i] / ((mixture.variances[gammas[i]]/4)/s2.current + 1) +
            m.current / (s2.current/(mixture.variances[gammas[i]]/4) + 1);
        
        sigma2.current = (1/s2.current + 1/(mixture.variances[gammas[i]]/4))^(-1);
        
        mus[i] = mu.current;
        sigma2s[i] = sigma2.current;
    }
    
    ## BACKWARD SAMPLER ##
    m.current = alpha*(1-theta) + theta*mu.current;
    s2.current = tau.square + theta^2*sigma2.current;
    
    h[n] = rnorm(n=1, mean = m.current, sd = sqrt(s2.current));
    
    for (i in seq(n-1,1)) {
        posterior.mean = ((h[i+1] - alpha*(1-theta))/theta) /
            ( (tau.square/theta^2)/sigma2s[i] + 1) +
            ## ##
            mus[i] /
            ( sigma2s[i]/(tau.square/theta^2) + 1);
        
        posterior.var = (1/sigma2s[i] + 1/(tau.square/theta^2))^(-1);
        
        h[i] = rnorm(n=1, mean = posterior.mean,
                     sd = sqrt(posterior.var));
    }
    return (h);
}

## Corresponds to section 2.4 in the document.
## Note on input: proposal.covariance is not needed in this
## function. I just wanted to keep its signiture the same as
## sample.phi.rw(...)
##
## The proposal distribution is a multivariate normal, with 1 degree
## of freedom. The proposal, and accept/reject step are performed on
## R^3 (what I'm calling tilde scale), following the transformations
## 
## alpha.tilde = alpha
## theta.tilde = logit(theta) [since theta \in (0,1)]
## tau.square.tilde = log(tau.square)
## 
## Here we follow the Chib approach of maximizing the likelihood and
## prior with respect to (alpha.tilde, theta.tilde, tau.square.tilde),
## using the maximum as the proposal mean and the negative inverse
## Hessian at the maximization point as a proposal covariance.
##
## Also, imporntantly, we INTEGRATE OUT THE LOG-VOLATILITIES to
## compute the likelihood.
sample.phi <- function(alpha,
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
                       delta.t) {
    n = length(h);

    ## Transforming current (alpha, theta, tau.square) parameters on
    ## tilde scale.
    theta.tilde.current = log(theta/(1-theta));
    tau.square.tilde.current = log(tau.square);
    phi.tilde.current = c(alpha, theta.tilde.current, tau.square.tilde.current);

    ## Maximizing the log-likelihood and prior over the tilde scale.
    min.phi.tilde <- optim(par = phi.tilde.current,
                           fn = log.likelihood.h.integrated.tilde.with.priors,
                           h = h,
                           y.star = y.star,
                           gammas = gammas,
                           mixture.means = mixture.means,
                           mixture.variances = mixture.variances,
                           alpha.prior.params = alpha.prior.params,
                           theta.prior.params = theta.prior.params,
                           tau.square.prior.params = tau.square.prior.params,
                           method = "Nelder-Mead",
                           control = list(fnscale = -1,
                                          reltol = 1e-10),
                           hessian = TRUE);

    print(min.phi.tilde$par);
    ## The log-likelihood-based proposal covariance matrix is the
    ## negative inverse Hessian matrix of the log-likelihood +
    ## log-prior at the maximization point.

    ## Here, we may run into the issue that the Hessian at the point
    ## is not invertible. However, this is not an issue as long as we
    ## have enough data and/or delta.t is small enough when compared
    ## to 1/theta.hat.
    proposal.covariance.ll =
        -1*solve(min.phi.tilde$hessian);
    print(proposal.covariance.ll);

    ## Proposed (alpha.tilde, theta.tilde, tau.square.tilde) on the
    ## tilde (R^3) scale.
    phi.tilde.proposal = min.phi.tilde$par + rmvt(n = 1,
                                                  sigma = proposal.covariance.ll,
                                                  df = 1);

    ## OK  THIS IS  A BIT  OF A  CHEAT!!  I  am making  sure that  the
    ## proposed theta.tilde corresponds to  a timescale of inertia for
    ## the OU  process that  is between  1 min  and 1  day. Otherwise,
    ## transforming back to the  nominal scale leads theta.proposal to
    ## be a NaN  (since its an Inf/Inf). This corresponds  to having a
    ## truncated prior for theta.hat
    lb = -delta.t/(60*1000) - log(1-exp(-delta.t/(60*1000)));
    ub = -delta.t/(6.5*60*60*1000) - log(1-exp(-delta.t/(6.5*60*60*1000)));

    ## If theta.tilde.proposal breaks the bounds, resample until it
    ## does not.
    while (phi.tilde.proposal[2] > ub ||
           phi.tilde.proposal[2] < lb)
    {
        print("resampling");
        print(min.phi.tilde$par);
        
        phi.tilde.proposal = min.phi.tilde$par + rmvt(n = 1,
                                                      sigma = proposal.covariance.ll,
                                                      df = 1);
        alpha.proposal = phi.tilde.proposal[1];
        ## ##
        theta.proposal = exp(phi.tilde.proposal[2])/(exp(phi.tilde.proposal[2])+1);
        theta.hat.proposal = log(theta.proposal)/(-delta.t);
        ## ##
        tau.square.proposal = exp(phi.tilde.proposal[3]);
        timescale.proposed = abs(1/theta.hat.proposal);
    }

    ## Transforming back to the nominal scale.
    alpha.proposal = phi.tilde.proposal[1];

    theta.proposal = exp(phi.tilde.proposal[2])/(exp(phi.tilde.proposal[2])+1);
    theta.hat.proposal = log(theta.proposal)/(-delta.t);

    tau.square.proposal = exp(phi.tilde.proposal[3]);


    timescale.proposed = abs(1/theta.hat.proposal);    
    print(paste("theta.tilde.proposal = ", phi.tilde.proposal[2], sep = ""));
    print(paste("theta.hat.proposal = ", theta.hat.proposal, sep = ""));
    print(paste("timescale.proposed = ", timescale.proposed/60000, sep = ""));

    
    theta.tilde.proposal = phi.tilde.proposal[2];
    tau.square.tilde.proposal = phi.tilde.proposal[3];
    
    phi.current = c(alpha, theta, tau.square);
    phi.proposal = c(alpha.proposal, theta.proposal, tau.square.proposal);

    ## Log-likelihood + log-priors for the current phi parameter
    ## vector, on the tilde scale.
    ll.current.tilde = log.likelihood.h.integrated(phi.current,
                                                   h,
                                                   y.star,
                                                   gammas,
                                                   mixture.means,
                                                   mixture.variances) +
        ## priors ##
        dnorm(x = alpha,
              mean = alpha.prior.params[1],
              sd = sqrt(alpha.prior.params[2]),
              log=TRUE) +
        dnorm(x = theta,
              mean = theta.prior.params[1],
              sd = sqrt(theta.prior.params[2]),
              log=TRUE) +
        dgamma(1/tau.square,
               shape = tau.square.prior.params[1],
               rate = tau.square.prior.params[2],
               log=TRUE) +
        ## log jacobian
        theta.tilde.current - 2*log(exp(theta.tilde.current) + 1) +
        tau.square.tilde.current;

    ## Log-likelihood + log-priors for the proposed phi parameter
    ## vector, on the tilde scale.
    ll.proposal.tilde = log.likelihood.h.integrated(phi.proposal,
                                                    h,
                                                    y.star,
                                                    gammas,
                                                    mixture.means,
                                                    mixture.variances) +
        ## priors ##
        dnorm(x = alpha.proposal,
              mean = alpha.prior.params[1],
              sd = sqrt(alpha.prior.params[2]),
              log=TRUE) +
        dnorm(x = theta.proposal,
              mean = theta.prior.params[1],
              sd = sqrt(theta.prior.params[2]),
              log=TRUE) +
        dgamma(1/tau.square.proposal,
               shape = tau.square.prior.params[1],
               rate = tau.square.prior.params[2],
               log=TRUE) +
        ## log jacobian
        theta.tilde.proposal - 2*log(exp(theta.tilde.proposal) + 1) +
        tau.square.tilde.proposal;

    ## Log acceptance ratio. Numerator is the log-(likelihood+priors)
    ## for proposed phi on tilde scale plus proposal density for
    ## current phi.tilde vector. Denominator is is the
    ## log-(likelihood+priors) for the current phi on tilde scale plus
    ## proposal density for proposed phi.tilde vector.
    log.accept.ratio = min(0, (ll.proposal.tilde +
                               dmvt(x=(phi.tilde.current-min.phi.tilde$par),
                                    sigma=proposal.covariance.ll,
                                    df = 1,
                                    log=TRUE))
                           -(ll.current.tilde +
                             dmvt(x=(phi.tilde.proposal-min.phi.tilde$par),
                                      sigma=proposal.covariance.ll,
                                  df = 1,
                                  log=TRUE)));

    out = NULL;
    out$proposal.covariance.ll = proposal.covariance.ll;
    
    print(log.accept.ratio);
        if (log(runif(1)) < log.accept.ratio) {
            ## accept proposal ##
            out$phi <- phi.proposal;
            return (out);
        } else {
            ## reject proposal ##
            out$phi <- phi.current;
            return (out);
        }
    }

## Corresponds to section 2.4 in the document.
## The proposal distribution is a multivariate normal, with 1 degree
## of freedom. The proposal, and accept/reject step are performed on
## R^3 (what I'm calling tilde scale), following the transformations
## 
## alpha.tilde = alpha
## theta.tilde = logit(theta) [since theta \in (0,1)]
## tau.square.tilde = log(tau.square)
##
## Here we use a multivariate normal walk with proposal.covariance as
## the proposal covariance matrix.
##
## Also, imporntantly, we INTEGRATE OUT THE LOG-VOLATILITIES to
## compute the likelihood.
sample.phi.rw <- function(alpha,
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
                          delta.t) {

    ## Transforming current (alpha, theta, tau.square) parameters on
    ## tilde scale.
    theta.tilde.current = log(theta/(1-theta));
    tau.square.tilde.current = log(tau.square);
    phi.tilde.current = c(alpha, theta.tilde.current, tau.square.tilde.current);

    ## Proposed (alpha.tilde, theta.tilde, tau.square.tilde) on the
    ## tilde (R^3) scale.
    phi.tilde.proposal = phi.tilde.current + rmvt(n = 1,
                                                  sigma = proposal.covariance,
                                                  df = 1);
    
    ## OK  THIS IS  A BIT  OF A  CHEAT!!  I  am making  sure that  the
    ## proposed theta.tilde corresponds to  a timescale of inertia for
    ## the OU  process that  is between  1 min  and 1  day. Otherwise,
    ## transforming back to the  nominal scale leads theta.proposal to
    ## be a NaN  (since its an Inf/Inf). This corresponds  to having a
    ## truncated prior for theta.hat
    lb = -delta.t/(60*1000) - log(1-exp(-delta.t/(60*1000)));
    ub = -delta.t/(6.5*60*60*1000) - log(1-exp(-delta.t/(6.5*60*60*1000)));

    ## If theta.tilde.proposal breaks the bounds, resample until it
    ## does not.
    while (phi.tilde.proposal[2] > ub ||
           phi.tilde.proposal[2] < lb)
    {
        print("resampling");
        print(phi.tilde.current);
        
        phi.tilde.proposal = phi.tilde.current + rmvt(n = 1,
                                                      sigma = proposal.covariance,
                                                      df = 1);
        alpha.proposal = phi.tilde.proposal[1];
        ## ##
        theta.proposal = exp(phi.tilde.proposal[2])/(exp(phi.tilde.proposal[2])+1);
        theta.hat.proposal = log(theta.proposal)/(-delta.t);
        ## ##
        tau.square.proposal = exp(phi.tilde.proposal[3]);
        timescale.proposed = abs(1/theta.hat.proposal);
    }

    ## Transforming back to the nominal scale.
    alpha.proposal = phi.tilde.proposal[1];

    theta.proposal = exp(phi.tilde.proposal[2])/(exp(phi.tilde.proposal[2])+1);
    theta.hat.proposal = log(theta.proposal)/(-delta.t);

    tau.square.proposal = exp(phi.tilde.proposal[3]);

    print(paste("theta.tilde.proposal = ", phi.tilde.proposal[2], sep = ""));
    print(paste("theta.hat.proposal = ", theta.hat.proposal, sep = ""));
    
    timescale.proposed = abs(1/theta.hat.proposal);
    
    print(paste("theta.tilde.proposal = ", phi.tilde.proposal[2], sep = ""));
    print(paste("theta.hat.proposal = ", theta.hat.proposal, sep = ""));
    print(paste("timescale.proposed = ", timescale.proposed/60000, sep = ""));
    
    theta.tilde.proposal = phi.tilde.proposal[2];
    tau.square.tilde.proposal = phi.tilde.proposal[3];
    
    phi.current = c(alpha, theta, tau.square);
    phi.proposal = c(alpha.proposal, theta.proposal, tau.square.proposal);

    ## Log-likelihood + log-priors for the current phi parameter
    ## vector, on the tilde scale.
    ll.current.tilde = log.likelihood.h.integrated(phi.current,
                                                   h,
                                                   y.star,
                                                   gammas,
                                                   mixture.means,
                                                   mixture.variances) +
        ## priors ##
        dnorm(x = phi.current[1],
              mean = alpha.prior.params[1],
              sd = sqrt(alpha.prior.params[2]),
              log=TRUE) +
        dnorm(x = phi.current[2],
              mean = theta.prior.params[1],
              sd = sqrt(theta.prior.params[2]),
              log=TRUE) +
        dgamma(1/phi.current[3],
               shape = tau.square.prior.params[1],
               rate = tau.square.prior.params[2],
               log=TRUE) +
        ## log jacobian
        theta.tilde.current - 2*log(exp(theta.tilde.current) + 1) +
        tau.square.tilde.current;

    ## Log-likelihood + log-priors for the proposed phi parameter
    ## vector, on the tilde scale.
    ll.proposal.tilde = log.likelihood.h.integrated(phi.proposal,
                                                    h,
                                                    y.star,
                                                    gammas,
                                                    mixture.means,
                                                    mixture.variances) +
        ## priors ##
        dnorm(x = alpha.proposal,
              mean = alpha.prior.params[1],
              sd = sqrt(alpha.prior.params[2]),
              log=TRUE) +
        dnorm(x = theta.proposal,
              mean = theta.prior.params[1],
              sd = sqrt(theta.prior.params[2]),
              log=TRUE) +
        dgamma(1/tau.square.proposal,
               shape = tau.square.prior.params[1],
               rate = tau.square.prior.params[2],
               log=TRUE) +
        ## log jacobian
        theta.tilde.proposal - 2*log(exp(theta.tilde.proposal) + 1) +
        tau.square.tilde.proposal;

    ## Log acceptance ratio ##
    log.accept.ratio = min(0, (ll.proposal.tilde-ll.current.tilde));
    
    if (log(runif(1)) < log.accept.ratio) {
        ## accept proposal ##
        return (phi.proposal);
    } else {
        ## reject proposal ##
        return (phi.current);
    }
}

## Log-likelihood for (y^* | phi), where we have integrated out h, all
## on the tilde scale.  Follows equations (9), (10), and (11).
log.likelihood.h.integrated.tilde <- function(phi.tilde,
                                              h,
                                              y.star,
                                              gammas,
                                              mixture.means,
                                              mixture.variances) {
    alpha.tilde = phi.tilde[1];
    theta.tilde = phi.tilde[2];
    tau.square.tilde = phi.tilde[3];
    
    alpha.nominal = alpha.tilde;
    theta.nominal = exp(theta.tilde)/(exp(theta.tilde) + 1);
    tau.square.nominal = exp(tau.square.tilde);
    
    out = log.likelihood.h.integrated(phi = c(alpha.nominal,
                                              theta.nominal,
                                              tau.square.nominal),
                                      h = h,
                                      y.star = y.star,
                                      gammas = gammas,
                                      mixture.means = mixture.means,
                                      mixture.variances = mixture.variances) +
        ## log-jacobian
        theta.tilde - 2*log(exp(theta.tilde) + 1) +
        tau.square.tilde;
    return(out);
}

## Log likelihood for p(y^* | phi), with h integrated, plus
## log-density for the priors p(phi).
log.likelihood.h.integrated.tilde.with.priors <- function(phi.tilde,
                                                          h,
                                                          y.star,
                                                          gammas,
                                                          mixture.means,
                                                          mixture.variances,
                                                          alpha.prior.params,
                                                          theta.prior.params,
                                                          tau.square.prior.params) {
    alpha.tilde = phi.tilde[1];
    theta.tilde = phi.tilde[2];
    tau.square.tilde = phi.tilde[3];
    
    alpha.nominal = alpha.tilde;
    theta.nominal = exp(theta.tilde)/(exp(theta.tilde) + 1);
    tau.square.nominal = exp(tau.square.tilde);
    
    out = log.likelihood.h.integrated(phi = c(alpha.nominal,
                                              theta.nominal,
                                              tau.square.nominal),
                                      h = h,
                                      y.star = y.star,
                                      gammas = gammas,
                                      mixture.means = mixture.means,
                                      mixture.variances = mixture.variances) +
        ## priors ##
        dnorm(x = alpha.nominal,
              mean = alpha.prior.params[1],
              sd = sqrt(alpha.prior.params[2]),
              log=TRUE) +
        dnorm(x = theta.nominal,
              mean = theta.prior.params[1],
              sd = sqrt(theta.prior.params[2]),
              log=TRUE) +
        dgamma(1/tau.square.nominal,
               shape = tau.square.prior.params[1],
               rate = tau.square.prior.params[2],
               log=TRUE) +
        ## jacobian ##
        theta.tilde - 2*log(exp(theta.tilde) + 1) +
        tau.square.tilde;
    return(out);
}

## Log-likelihood for (y^* | phi), where we have integrated out h, all
## on the nominal scale.  Follows equations (9), (10), and (11).
log.likelihood.h.integrated <- function(phi,
                                        h,
                                        y.star,
                                        gammas,
                                        mixture.means,
                                        mixture.variances) {
    alpha = phi[1];
    theta = phi[2];
    tau.square = phi[3];
    
    mu.current = alpha;
    sigma2.current = tau.square / (1-theta);

    n = length(h);
    mus = rep(NA, n-1);
    sigma2s = rep(NA, n-1);
    
    ll = 0;
    ## FORWARD FILTER ##
    for (i in seq(1,n-1)) {
        m.current = alpha*(1-theta) + theta*mu.current;
        s2.current = tau.square + theta^2*sigma2.current;
        
        ll = ll +
            dnorm(x = y.star[i],
                  mean = mixture.means[gammas[i]]/2 + m.current,
                  sd = sqrt(mixture.variances[gammas[i]]/4 + s2.current),
                  log = TRUE);
        
        mu.current =
            y.star[i] / ((mixture.variances[gammas[i]]/4)/s2.current + 1) +
            m.current / (s2.current/(mixture.variances[gammas[i]]/4) + 1);
        
        sigma2.current = (1/s2.current + 1/(mixture.variances[gammas[i]]/4))^(-1);
        
        mus[i] = mu.current;
        sigma2s[i] = sigma2.current; 
    }
    
    ## for (i in seq(1,n-1)) {
    ##     ll = ll +
    ##         dnorm(x = h[i+1],
    ##               mean = phi[1] + phi[2]*(h[i]-phi[1]),
    ##               sd = sqrt(phi[3]),
    ##               log = TRUE);
    ## }
    return (ll);
}

## Follows section 2.1 in the document.
sample.latent.prices <- function(log.prices,
                                 xi.2,
                                 h) {

    n = length(log.prices);
    log.latent.prices = rep(NA, n);
    mu.current = log.prices[1];
    sigma2.current = xi.2;
    
    mus = rep(NA, length(log.prices));
    sigma2s = rep(NA, length(log.prices));
    
    ## FORWARD FILTER ##
    for (i in seq(1,length(log.prices))) {
        m.current = mu.current;
        s2.current = sigma2.current + exp(2*h[i]);
        
        mu.current =
            log.prices[i] / ( exp(log(xi.2)-log(s2.current)) + 1) +
            m.current / ( exp(log(s2.current)-log(xi.2)) + 1);
        
        sigma2.current = exp(
            log(xi.2) + log(s2.current) -
            log(xi.2 + s2.current));
        
        mus[i] = mu.current;
        sigma2s[i] = sigma2.current;
    }
    
    ## BACKWARD SAMPLER ##
    log.latent.prices[length(log.latent.prices)] = rnorm(n=1,
                                                         mean = mus[n],
                                                         sd = sqrt(sigma2s[n]));
    for (i in seq(n-1,1)) { 
        posterior.mean =
            log.latent.prices[i+1] / (exp(2*h[i+1]-log(sigma2s[i])) + 1) +
            mus[i] / (exp(log(sigma2s[i])-2*h[i+1]) + 1);
        
        posterior.var = exp(
            log(sigma2s[i]) + 2*h[i+1] -
            log(sigma2s[i] + exp(2*h[i+1])));
        
        log.latent.prices[i] = rnorm(n=1,
                                     mean = posterior.mean,
                                     sd = sqrt(posterior.var));
    }
    return (log.latent.prices);
}
