This is an instruction manual for running MCMC on the simplified model
in ``MCMC-documentation-for-Newcode.pdf''. 

A. DATA GENERATION 

1. Open "generate-data.R". All of the parameters have been
   commented. You must change ``save.directory'' to a directory that
   is accessible by R and where you want to save the simulated data.

2. After editing "generate-data.R", in R run
   source("generate-data.R"). This will create five files in your
   ``save.directory'' directory, as documented in the
   generate.simulated.data(...) function in
   "generate-simulated-data.R" file. Most important of these are the
   *added-noise*.Rdata, *no-noise*.Rdata, and model-parameters*.Rdata
   files.

B. RUNNING MCMC 

1. After generating the data, open up "main.R", and change the
   ''data.file'' and ''model.parameters.file'' to the files containing
   the data you want to analyze and the model.parameters file
   containing the parameters used in the data generation.

2. run source("main.R").

 
