theta.prior.parameters <- function(theta.hat.mean,
                                   theta.hat.var,
                                   delta.t) {

    ## corresponds to equation (17) in the paper
    mean.lhs <- function(a, b) {
        out = a + (dnorm(-a/b) - dnorm((1-a)/b)) / (pnorm((1-a)/b) - pnorm(-a/b)) *
            b;
        return (out);
    }

    ## corresponds to equation (18) in the paper
    var.lhs <- function(a, b) {
        first.term = (-a/b*dnorm(-a/b) - (1-a)/b*dnorm((1-a)/b))/
            (pnorm((1-a)/b) - pnorm(-a/b));
        second.term = (dnorm(-a/b) - dnorm((1-a)/b))^2/(pnorm((1-a)/b) - pnorm(-a/b))^2;

        out = b^2 * (1 + first.term + second.term);
        return (out);
    }

    theta.prior.params.minimization <- function(x,
                                                theta.hat.mean,
                                                theta.hat.var,
                                                delta.t) {
        a = x[1];
        b = exp(x[2]);

        mu.lhs <- mean.lhs(a,b);
        sigma2.lhs <- var.lhs(a,b);

        ## equation (19) in paper
        first.moment.rhs = exp(-theta.hat.mean*delta.t)*
            (1 + 0.5*theta.hat.var*delta.t^2);
        ## equation (20) in paper
        second.moment.rhs = exp(-2*theta.hat.mean*delta.t)*
            (1 + 2*theta.hat.var*delta.t^2);
        
        mu.rhs = first.moment.rhs;
        sigma2.rhs = second.moment.rhs - first.moment.rhs^2;

        out = (mu.lhs - mu.rhs)^2 + (sigma2.rhs - sigma2.lhs)^2
        return (out);
    }

    ## starting values ##
    x = c(theta.hat.mean*delta.t, log(sqrt(theta.hat.var) * delta.t));

    a.log.b = optim(par = x,
                    fn = theta.prior.params.minimization,
                    theta.hat.mean = theta.hat.mean,
                    theta.hat.var = theta.hat.var,
                    delta.t = delta.t,
                    method = "Nelder-Mead");

    theta.mean = a.log.b$par[1];
    theta.std.dev = exp(a.log.b$par[2]);

    return (c(theta.mean, theta.std.dev^2));
}

tau.square.prior.parameters <- function(tau.square.hat.mean,
                                        tau.square.hat.var,
                                        theta.hat.mean,
                                        theta.hat.var,
                                        delta.t) {

    ## corresponds to right hand side of equations (21) and (22) in the paper
    first.second.moment.rhs <- function(tau.square.hat.mean,
                                        tau.square.hat.var,
                                        theta.hat.mean,
                                        theta.hat.var,
                                        delta.t) {
        
        f = 0.5 *
            tau.square.hat.mean *
            (1.0-exp(-2.0*theta.hat.mean*delta.t))/theta.hat.mean;
        
        f.prime.theta = 0.5 *
            tau.square.hat.mean *
            (2.0*delta.t*exp(-2.0*theta.hat.mean*delta.t)/theta.hat.mean - 
             (1.0-exp(-2.0*theta.hat.mean*delta.t))/(theta.hat.mean^2));
        
        f.prime.tau.sq = 0.5 *
            (1.0-exp(-2.0*theta.hat.mean*delta.t))/theta.hat.mean;
        
        f.double.prime.theta = 0.5 *
            tau.square.hat.mean *
            (-4.0* (delta.t^2) *exp(-2.0*theta.hat.mean*delta.t)/theta.hat.mean
                - 2.0*2.0*delta.t*exp(-2.0*theta.hat.mean*delta.t)/(theta.hat.mean)^2
                + 2.0*(1-exp(-2.0*theta.hat.mean*delta.t))/(theta.hat.mean)^3);
        
        f.double.prime.tau.sq = 0.0;

        first.moment.first.term = f;

        first.moment.second.term = 
            0.5 * f.double.prime.tau.sq * (tau.square.hat.var);
        
        first.moment.third.term = 
            0.5 * f.double.prime.theta * (theta.hat.var);
        
        second.moment.first.term = 
            (f^2);
        
        second.moment.second.term = 
                   0.5 *
        (2.0*(f.prime.theta^2) + 2.0*f*f.double.prime.theta) *
        (theta.hat.var);
        
        second.moment.third.term = 
                   0.5 *
                   (2.0*(f.prime.tau.sq)^2 + 2.0*f*f.double.prime.tau.sq) *
                   (tau.square.hat.var);

        ## corresponds to equation (21)
        first.moment.rhs = 
            first.moment.first.term + 
            first.moment.second.term +
            first.moment.third.term;

        ## corresponds to equation (22)
        second.moment.rhs = 
            second.moment.first.term + 
            second.moment.second.term + 
            second.moment.third.term;

        return (c(first.moment.rhs, second.moment.rhs));
    }

    tau.square.prior.params.minimization <- function(x,
                                                     tau.square.hat.mean,
                                                     tau.square.hat.var,
                                                     theta.hat.mean,
                                                     theta.hat.var,
                                                     delta.t,
                                                     first.moment.rhs,
                                                     second.moment.rhs) {

        ## lower bound on alpha is 2 to ensure second moment exists
        alpha = 2 + exp(x[1]);
        beta = exp(x[2]);

        ## the mean and variance of tau^2(\Delta) based on the
        ## Inv-Gamma prior for tau^2(\Delta)
        mu.lhs <- beta/(alpha-1);
        sigma2.lhs <- beta^2 / ((alpha-1)^2*(alpha-2));

        mu.rhs = first.moment.rhs;
        sigma2.rhs = second.moment.rhs - first.moment.rhs^2;

        out = (mu.lhs - mu.rhs)^2 + (sigma2.rhs - sigma2.lhs)^2
        return (out);
    }

    first.second.moments.rhs <- first.second.moment.rhs(tau.square.hat.mean,
                                                        tau.square.hat.var,
                                                        theta.hat.mean,
                                                        theta.hat.var,
                                                        delta.t);
    first.moment.rhs = first.second.moments.rhs[1];
    second.moment.rhs = first.second.moments.rhs[2];    
    
    first.moment.approx = tau.square.hat.mean * 
      (1.0-exp(-2.0*theta.hat.mean*delta.t)) / 
      (2.0*theta.hat.mean);

    mean.approx = first.moment.approx;
    variance.approx = (first.moment.approx * 10)^2;

    alpha.start = (mean.approx)^2 / variance.approx + 2;
    alpha.start = max(c(2+1e-16, alpha.start));
    beta.start = mean.approx * (alpha.start-1);

    x = c(log(alpha.start-2), log(beta.start));

    tau.square.prior.alpha.beta <- optim(par = x,
                                         fn = tau.square.prior.params.minimization,
                                         tau.square.hat.mean = tau.square.hat.mean,
                                         tau.square.hat.var = tau.square.hat.var,
                                         theta.hat.mean = theta.hat.mean,
                                         theta.hat.var = theta.hat.var,
                                         delta.t = delta.t,
                                         first.moment.rhs = first.moment.rhs,
                                         second.moment.rhs = second.moment.rhs);
    
    alpha = exp(tau.square.prior.alpha.beta$par[1]) + 2;
    beta = exp(tau.square.prior.alpha.beta$par[2])

    return(c(alpha,beta));
}
