#include <torch/torch.h>
#include <torch/extension.h>
#include <Eigen/Core>
#include <vector>
#include "factorizers.h"


namespace rlap {

    Eigen::MatrixXd tensorToEigen(const at::Tensor& tensor) {
        // Ensure the tensor is of the correct type and dimension
        assert(tensor.dtype() == torch::kDouble);
        assert(tensor.dim() == 2);

        // Extract dimensions
        int64_t rows = tensor.size(0);
        int64_t cols = tensor.size(1);

        // Get a pointer to the tensor data
        double* tensor_data = tensor.data_ptr<double>();

        // Map the tensor data to an Eigen::Map object with row-major storage
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eigen_matrix(tensor_data, rows, cols);

        // As per the docs: https://eigen.tuxfamily.org/dox/group__TopicStorageOrders.html
        // Matrices and arrays using one storage order can be assigned to matrices and arrays using the other storage order.
        // Eigen will reorder the entries automatically. More generally, row-major and column-major matrices can be mixed in an expression as we want.
        // Thus, by ensuring that the pytorch tensor data is properly copied to the Eigen tensor, we are good to proceed with any Eigen related operations.

        return eigen_matrix;
    }

    at::Tensor eigenToTensor(const Eigen::MatrixXd& eigen_matrix) {
        // Get the dimensions of the Eigen matrix
        int64_t rows = eigen_matrix.rows();
        int64_t cols = eigen_matrix.cols();

        // get row-major storage
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_major_matrix = eigen_matrix;

        // Create a tensor of the same shape
        at::Tensor tensor = torch::empty({rows, cols}, torch::kDouble);

        // Get a pointer to the tensor data
        double* tensor_data = tensor.data_ptr<double>();

        // Copy data from the row-major Eigen matrix to the tensor
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(tensor_data, rows, cols) = row_major_matrix;

        return tensor;
    }


    at::Tensor approximate_cholesky_cpu(const at::Tensor& edge_info, const int64_t num_nodes, const int64_t num_remove,
                                        const std::string o_v,  const std::string o_n) {
        TORCH_INTERNAL_ASSERT(edge_info.device().type() == at::DeviceType::CPU);
        ApproximateCholesky ac = ApproximateCholesky();
        Eigen::MatrixXd eigen_edge_info = tensorToEigen(edge_info.contiguous());
        ac.setup(
            /*edge_info=*/eigen_edge_info,
            /*nrows=*/num_nodes,
            /*ncols=*/num_nodes,
            /*o_v=*/o_v,
            /*o_n=*/o_n
        );
        Eigen::MatrixXd eigen_result = ac.getSchurComplement(/*t=*/num_remove);
        at::Tensor result = eigenToTensor(eigen_result);
        return result;
    }

    at::Tensor identity_cpu(const at::Tensor& a){
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
        Eigen::MatrixXd eigen_a = tensorToEigen(a.contiguous());
        at::Tensor torch_a = eigenToTensor(eigen_a);
        return torch_a;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(extension_cpp, m) {
        m.def("approximate_cholesky(Tensor edge_info, int num_nodes, int num_remove, str o_v,  str o_n) -> Tensor");
        m.def("identity(Tensor a) -> Tensor");
    }

    TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
        m.impl("approximate_cholesky", &approximate_cholesky_cpu);
        m.impl("identity", &identity_cpu);
    }
}
