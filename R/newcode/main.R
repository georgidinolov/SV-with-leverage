## main.R
## This is a driver for running MCMC on the simplified model described
## in ``MCMC Documentation for Newcode.''
rm(list=ls());
library(sendmailR);

## file location of where the data set has been saved
data.file = "/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/simulated-data/theta-9e+05/simulation-1/simulated-prices-and-returns-added-noise-9-22-16-39-11.Rdata";

## file location of where the model parameter object used to generate
## the data has been saved
model.parameters.file = "/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/simulated-data/theta-9e+05/simulation-1/model-parameters-9-22-16-39-11.Rdata";

## directory where we save the posterior draws
save.directory =
    "/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/simulated-data/theta-9e+05/simulation-1/";

## SET DELTA.T ##
delta.t = 1*15*1000;
##  ## 

source("/share/Arbeit/gdinolov/SV-with-leverage/R/newcode/sv-single-factor-no-rho.R");
proposal.covariance = 2*matrix(nrow = 3, ncol = 3,
                                   data = c(c(0.0005969138, -0.00193,     0.0007709064),
                                            c(-0.00193,      0.02484804, -0.0228),
                                            c(0.0007709064, -0.0228,      0.0372519808)),
                                   byrow = TRUE);

input = NULL;
input$data.file = data.file;
input$model.parameters.file = model.parameters.file;
input$save.directory = save.directory;

posterior.results <- run.mcmc(load.save.input = input,
                              random.walk = FALSE,
                              xi.2.override = FALSE,
                              xi.2.set = 0,
                              proposal.covariance = proposal.covariance,
                              delta.t = delta.t,
                              number.posterior.samples = 10,
                              burn.in = 0);

from <- sprintf("<sendmailR@\\%s>", Sys.info()[4])
to <- "<gdinolov@soe.ucsc.edu>"
subject <- "Job complete"
body <- list("Job complete");
sendmail(from, to, subject, body,
         control=list(smtpServer="ASPMX.L.GOOGLE.COM"))