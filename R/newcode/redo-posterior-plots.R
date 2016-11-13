rm(list=ls());
library(MCMCpack);	
delta.t = 5000;

## The ``root.directory'' is the directory where the simulation
## folders for data sets are held
root.directory = paste(
    "/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/simulated-data/theta-",
    "9e+05", "/", sep = "");

## Check if directory exists.
if (file.exists(root.directory) == FALSE) {
    print(paste(root.directory, " doesn't exist! Can't load data.", sep = ""));
}

## Find all simulation data folders in the root directory.
data.directories =
    list.files(path=root.directory, pattern = "*simulation*");

posterior.results.list = vector(mode = "list",
                               length = length(data.directories));

for (i in seq(1,length(data.directories))) {
    print(i);
    posterior.file = list.files(path = paste(root.directory,
                                        data.directories[i],
                                        "/",	
                                        sep = ""),
                           pattern = paste("*deltat-", delta.t, "*", sep = ""))[1];

    posterior.results.list[[i]] = paste(root.directory,
					data.directories[i],
                                        "/",	
					posterior.file,
                                        sep = "");
}

print(posterior.results.list);

number.covered.IVs = 0;
for (i in seq(1,length(posterior.results.list))) {
    load(posterior.results.list[[i]]);
    load(load.save.input$model.parameters.file);

    alpha.hats = out$alpha.hats;
    theta.hats = out$theta.hats;
    tau.square.hats = out$tau.square.hats;
    log.volatilities.quantiles = out$log.volatilities.quantiles;
    h.true = out$h.true;
    n = length(h.true);

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
         ## ylim = c(min(min(log.volatilities.quantiles[1,]), min(h.true)),
         ##          max(max(log.volatilities.quantiles[3,]), max(h.true))),
	 ylim = c(-11, -7),
         type = "l");
    coord.x = c(seq(1,n), seq(n,1));
    coord.y = c(log.volatilities.quantiles[3,seq(1,n)],
                log.volatilities.quantiles[1,seq(n,1)]);
    polygon(coord.x, coord.y, col = "grey");
    lines(log.volatilities.quantiles[2,], lwd = 2);
    lines(h.true, col = "red", lwd = 2);

    dev.off();

    print(i);
}

