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