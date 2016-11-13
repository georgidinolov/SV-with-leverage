rm(list=ls());
library(sendmailR);
library(parallel);
source("sv-single-factor-no-rho.R");

## change according to what your system can accomodate.
number.clusters = 96;

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

load.save.input.list = vector(mode = "list",
                              length = length(data.directories));

for (i in seq(1,length(data.directories))) {
    print(i);
    
    data.file = list.files(path = paste(root.directory,
                                        data.directories[i],
                                        "/",
                                        sep = ""),
                           pattern = "*added-noise*")[1];
    
    model.parameters.file = list.files(path = paste(root.directory,
                                        data.directories[i],
                                        "/",
                                        sep = ""),
                                       pattern = "*model-parameters*")[1];
    save.directory = paste(root.directory,
                           data.directories[i], "/",
                           sep = "");
    
    load.save.input.list[[i]]$data.file =
        paste(root.directory, data.directories[i],
              "/",
              data.file,
              sep = "");
    
    load.save.input.list[[i]]$model.parameters.file =
        paste(root.directory,
              data.directories[i],
              "/",
              model.parameters.file,
              sep = "");
    
    load.save.input.list[[i]]$save.directory =
        paste(root.directory,
              data.directories[i],
              "/",
              sep = "");
}

## proposal covariances have been found by questionable methods.
delta.ts <- c(15*1000);
proposal.covariances <- vector(mode = "list", length = length(delta.ts))
proposal.covariances[[1]] = matrix(nrow = 3, ncol = 3,
                                   data = c(c(0.113516350, 0.005157707, 0.004630331),
                                            c(0.005157707, 0.256452754, 0.004690798),
                                            c(0.004630331, 0.004690798, 0.034875728)),
                                   byrow = TRUE);
proposal.covariances[[2]] = 3*matrix(nrow = 3, ncol = 3,
                                   data = c(c(0.0005, -0.002,  0.0008808469),
                                            c(-0.002,  0.033541754, -0.027),
                                            c(0.0008808469, -0.027,  0.0475597352)),
                                   byrow = TRUE);
proposal.covariances[[3]] = 3*matrix(nrow = 3, ncol = 3,
                                   data = c(c(0.0005969138, -0.00193,     0.0007709064),
                                            c(-0.00193,      0.02484804, -0.0228),
                                            c(0.0007709064, -0.0228,      0.0372519808)),
                                   byrow = TRUE);

for (delta.t in delta.ts) {	   
    cl <- makeCluster(number.clusters);
    output.list <- parLapply(cl = cl,
                             X = load.save.input.list,
                             fun = run.mcmc,
                             random.walk = FALSE,
                             xi.2.override = FALSE,
                             xi.2.set = 0,
                             proposal.covariance =
                                 proposal.covariances[[which(delta.ts==delta.t)]],
                             delta.t = delta.t,
                             number.posterior.samples = 2,
                             burn.in = 0);
    stopCluster(cl);
}

from <- sprintf("<main-parallel@\\%s>", Sys.info()[4])
to <- "<gdinolov@soe.ucsc.edu>"
subject <- "Job complete"
body <- list("Job complete");
sendmail(from, to, subject, body,
         control=list(smtpServer="ASPMX.L.GOOGLE.COM"))
