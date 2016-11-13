rm(list = ls());
source("generate-simulated-data.R");

## ##### PARAMETERS FOR SIMULATION START (ie: true data-generating parameters) ##### ## 
## The below parameters correspond to equation (4) in the document.
alpha.hat = -13;
mean.reversion.timescale = 15*60*1000; ## mean-reversion timescale for
                                       ## the AR(1) process; here I'm
                                       ## setting it to 15 min
theta.hat = 1/mean.reversion.timescale; ## inverse timescale for the
                                        ## AR(1) latent state
VV = 0.116; ## VVX on the continuous log-volatility
            ## [log(\hat{sigma_t}) in the paper] scale. VVX is the vol
            ## of vol index.
tau.square.hat = VV * 2*theta.hat;
xi.2 = 6.5e-7;
TT = 6.5*60*60*1000; ## observational duration of a single trading day (6.5 hrs).

number.clusters = 10; ## when generating multiple data sets, increase
                      ## the number of clusters to be used by R.
## ##### PARAMETERS FOR SIMULATION END ##### ##

## MODEL PARAMETERS START ##
## model.parameters is an object containing the continuous-time
## parameters for the model, as well as the observational duration
## (set above to a single trading day in milliseconds), and the data
## generating step [$\Delta_{gen}$ in the paper].
model.parameters = NULL;
model.parameters$alpha.hat = alpha.hat;
model.parameters$tau.square.hat = tau.square.hat;
model.parameters$theta.hat = theta.hat;
model.parameters$xi.square = xi.2;
model.parameters$TT = TT; ## observational duration
model.parameters$delta.t.generation = 5; ## data generating step. 5
                                         ## milliseconds is assumed to
                                         ## be small enough to
                                         ## accurately approximate
                                         ## sampling from the
                                         ## continuous-time
                                         ## model. This can be changed
                                         ## to an integer number as
                                         ## small as 1.  MODEL
## MODEL PARAMETERS END ##

## We will generate N.data.sets simulated data sets, each in its own directory.
N.data.sets = 1;

## The ``root.directory'' is the directory where the simulation
## folders for data sets are held. Right now, we will store each data
## set in a directory for the specified timescale of inertia.
root.directory = paste("/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/simulated-data/theta-",
                       round(1/model.parameters$theta.hat), "/", sep = "");

## If the root doesn't exist, create one.
if (file.exists(root.directory) == FALSE) {
    print(paste(root.directory, " doesn't exist! Creating one", sep = ""));
    dir.create(root.directory, showWarnings = FALSE);
}

## Create the list of directories where the data sets will be saved,
## each labelled ``simulation-#''
save.path.list = vector(mode = "list",
                        length = N.data.sets);
for (i in seq(1,N.data.sets)) {
    save.path.list[[i]] = 
        paste(root.directory, "simulation-", i, "/", sep = "");
}

## Generate the multiple data sets.
generate.multiple.simulated.data(model.parameters = model.parameters,
                                 save.path.list = save.path.list,
                                 number.clusters = number.clusters);
