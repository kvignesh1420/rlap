load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "_librlap",
    srcs = ["py_api_binder.cc"],
    deps = [
        "//rlap/cc:factorizers",
        "//third_party/eigen3"
    ]
)