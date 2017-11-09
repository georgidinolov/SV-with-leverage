cc_binary(
	name = "georgi-test",
	srcs = ["georgi-test.cpp"],
	includes = ["nlopt.hpp"],
	linkopts = ["-lm"],
	deps = [":nlopt"],
	copts = ["-Isrc/nlopt/api",
		 "-O"],
)

cc_library(
	name = "nlopt",
	srcs = [".libs/libnlopt.a"],
	hdrs = glob(["api/nlopt*.h*"]),
	linkopts = ["-lm"],
	visibility = ["//visibility:public"],	
	copts = ["-Isrc/nlopt/api"],
)     
