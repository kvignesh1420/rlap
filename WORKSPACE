workspace(name = "rlap")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

EIGEN_COMMIT = "b3bea43a2da484d420e20c615cb5c9e3c04024e5"
EIGEN_SHA256 = "ffc9e46125c12c84422a477deacb8d36e1939461146427d1f38d3ded112af1da"

http_archive(
    name = "eigen_archive",
    build_file = "//third_party/eigen3:eigen_archive.BUILD",
    sha256 = EIGEN_SHA256,
    strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
    ],
)

PB_COMMIT = "26973c0ff320cb4b39e45bc3e4297b82bc3a6c09"
PB_SHA256 = "8f546c03bdd55d0e88cb491ddfbabe5aeb087f87de2fbf441391d70483affe39"

http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-{commit}".format(commit = PB_COMMIT),
    sha256 = PB_SHA256,
    urls = [
        "http://mirror.tensorflow.org/github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.tar.gz",
        "https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.tar.gz",
    ],
)

http_archive(
    name = "pybind11",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.6.0.tar.gz",
        "https://github.com/pybind/pybind11/archive/v2.6.0.tar.gz",
    ],
    sha256 = "90b705137b69ee3b5fc655eaca66d0dc9862ea1759226f7ccd3098425ae69571",
    strip_prefix = "pybind11-2.6.0",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")