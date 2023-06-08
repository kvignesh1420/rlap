cc_library(
    name = "types",
    hdrs = ["types.h"],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "tracer",
    hdrs = ["tracer.h"]
)

cc_library(
    name = "reader",
    srcs = ["reader.cc"],
    hdrs = ["reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3"
    ]
)

cc_library(
    name = "factorizers",
    srcs = ["factorizers.cc"],
    hdrs = ["factorizers.h"],
    deps = [
        ":types",
        ":reader",
        ":preconditioner",
        ":tracer",
        "//third_party/eigen3"
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "preconditioner",
    srcs = ["preconditioner.cc"],
    hdrs = ["preconditioner.h"],
    deps = [
        ":types",
        ":tracer",
        "//third_party/eigen3"
    ],
    visibility = ["//visibility:public"]
)
