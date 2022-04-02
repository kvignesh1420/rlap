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
    name = "samplers",
    srcs = ["samplers.cc"],
    hdrs = ["samplers.h"],
    deps = [
        "//third_party/eigen3"
    ]
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
        ":samplers",
        ":cg",
        ":reader",
        "//third_party/eigen3"
    ],
    visibility = ["//visibility:public"]
)

cc_library(
    name = "cg",
    srcs = ["cg.cc"],
    hdrs = ["cg.h"],
    deps = [
        ":types",
        "//third_party/eigen3"
    ],
    visibility = ["//visibility:public"]
)

