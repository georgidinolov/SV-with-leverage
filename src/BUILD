cc_binary(
	name = "data-types-test",
	srcs = ["data-types-test.cpp"],
	includes = ["DataTypes.hpp"],
	deps = [":data-types",],
)

cc_binary(
	name = "param-types-test",
	srcs = ["param-types-test.cpp"],
	includes = ["ParamTypes.hpp",
		    "DataTypes.hpp"],
	deps = [":param-types", ":data-types"],
)

cc_binary(
	name = "prior-types-test",
	srcs = ["prior-types-test.cpp"],
	includes = ["PriorTypes.hpp"],
	deps = [":prior-types",],
)

cc_binary(
	name = "proposal-types-test",
	srcs = ["proposal-types-test.cpp"],
	includes = ["ProposalTypes.hpp"],
	deps = [":proposal-types",],
)

cc_binary(
	name = "stochastic-vol-model-test",
	srcs = ["stochastic-vol-model-test.cpp"],
	deps = [":stochastic-vol-model"],
	copts = ["-O3"],
)

cc_binary(
	name = "observational-model-sampler-test",
	srcs = ["observational-model-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "constant-volatility-sampler-test",
	srcs = ["constant-volatility-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "ou-sampler-test",
	srcs = ["ou-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "stochastic-vol-sampler-test",
	srcs = ["stochastic-vol-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "multifactor-stochastic-vol-sampler-test",
	srcs = ["multifactor-stochastic-vol-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "multifactor-stochastic-vol-with-jumps-sampler-test",
	srcs = ["multifactor-stochastic-vol-with-jumps-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
	copts = ["-O3"],
)

cc_binary(
	name = "multifactor-stochastic-vol-with-jumps-sampler-test-fixed-rho",
	srcs = ["multifactor-stochastic-vol-with-jumps-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "multifactor-stochastic-vol-with-jumps-sampler-test-switched-rho",
	srcs = ["multifactor-stochastic-vol-with-jumps-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "multifactor-stochastic-vol-with-jumps-sampler-test-tau-fast-zero",
	srcs = ["multifactor-stochastic-vol-with-jumps-sampler-test.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "multifactor-stochastic-vol-with-jumps-sampler-real-data",
	srcs = ["multifactor-stochastic-vol-with-jumps-sampler-real-data.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "stochastic-vol-data",
	srcs = ["stochastic-vol-data.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "multifactor-stochastic-vol-data",
	srcs = ["multifactor-stochastic-vol-data.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "multifactor-stochastic-vol-with-jumps-data",
	srcs = ["multifactor-stochastic-vol-with-jumps-data.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler",
	        "//src/armadillo-7.600.2:armadillo"],
	copts = ["-O3",
		 "-Isrc/armadillo-7.600.2/usr/include"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_binary(
	name = "multifactor-stochastic-vol-with-jumps-rho-ll",
	srcs = ["multifactor-stochastic-vol-with-jumps-rho-ll.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas", "-larmadillo"],
)

cc_binary(
	name = "multifactor-stochastic-vol-with-jumps-theta-ll",
	srcs = ["multifactor-stochastic-vol-with-jumps-theta-ll.cpp"],
	includes = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":stochastic-vol-posterior-sampler"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
)

cc_library(
	name = "data-types",
	srcs = ["DataTypes.cpp"],
	hdrs = ["DataTypes.hpp"],
	visibility = ["//visibility:public"],
	copts = ["-O3"],
)

cc_library(
	name = "param-types",
	srcs = ["ParamTypes.cpp"],
	hdrs = ["ParamTypes.hpp"],
	visibility = ["//visibility:public"],
	linkopts = ["-lm"],
	copts = ["-O3"],
)

cc_library(
	name = "prior-types",
	srcs = ["PriorTypes.cpp"],
	hdrs = ["PriorTypes.hpp"],
	deps = ["//src/nlopt:nlopt"],
	linkopts = ["-lm", "-lRmath"],
	visibility = ["//visibility:public"],
	copts = ["-O3"],
)

cc_library(
	name = "proposal-types",
	srcs = ["ProposalTypes.cpp"],
	hdrs = ["ProposalTypes.hpp"],
	deps = [":multivariate-normal"],
 	linkopts = ["-lm", "-lgsl", "-lgslcblas"],
	visibility = ["//visibility:public"],
	copts = ["-O3"],	
)

cc_library(
	name = "multivariate-normal",
	srcs = ["MultivariateNormal.cpp"],
	hdrs = ["MultivariateNormal.hpp"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas"],
	visibility = ["//visibility:public"],
	deps = ["//src/armadillo-7.600.2:armadillo"],
	copts = ["-O3"],
)

cc_library(
	name = "stochastic-vol-model",
	srcs = ["StochasticVolatilityModel.cpp"],
	hdrs = ["StochasticVolatilityModel.hpp"],
	deps = [":data-types",
	        ":param-types",
		":prior-types",
	        ":multivariate-normal",
		"//src/armadillo-7.600.2:armadillo",
		"//src/nlopt:nlopt"],
	visibility = ["//visibility:public"],
	linkopts = ["-lm", 
		    "-lRmath",
		    "-lgsl", 
		    "-lgslcblas"],
	copts = ["-O3"],
)

cc_library(
	name = "stochastic-vol-posterior-sampler",
	srcs = ["StochasticVolatilityPosteriorSampler.cpp"],
	hdrs = ["StochasticVolatilityPosteriorSampler.hpp"],
	deps = [":data-types", 
	        ":param-types",
	        ":prior-types", 
		":proposal-types",
		":stochastic-vol-model",
 		":multivariate-normal",
		"//src/armadillo-7.600.2:armadillo"],
	copts = ["-O3"],
	linkopts = ["-lm", "-lRmath", "-lgsl", "-lgslcblas"],
	visibility = ["//visibility:public"],
)