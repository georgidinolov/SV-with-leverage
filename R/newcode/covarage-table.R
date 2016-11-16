rm(list=ls());
delta.t = 15000;

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

    qqs = quantile(x = out$IVs, probs = c(0.025, 0.975));

    if ( ((IV.true < qqs[1]) || (IV.true > qqs[2])) == FALSE ) {
       number.covered.IVs = number.covered.IVs + 1;
       print(posterior.results.list[[i]]);      
    } else {

    }     

}

print(paste("Fraction of covered IVs: ", 
		      number.covered.IVs / length(posterior.results.list), 
		      sep = ""));
