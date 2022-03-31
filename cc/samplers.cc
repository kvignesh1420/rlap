#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <functional>
#include "samplers.h"

Eigen::MatrixXf ExactCliqueSampler::sampleClique(Eigen::SparseMatrix<float>& L, int i){
    std::vector<int> N_i;
    for(int k = 0; k < L.cols(); k++){
        if(i!=k && std::abs(L.coeff(i, k)) >= 1e-8 ){
            N_i.push_back(k);
        }
    }
    // std::cout << "Node " << i << " has neighbour count = "<< N_i.size() << std::endl;
    Eigen::SparseMatrix<float> D_ts(L.rows(), L.cols()*(L.cols()-1));
    D_ts.reserve(Eigen::VectorXi::Constant(L.cols()*(L.cols()-1), 2));
    int edge_idx = 0;
    for(int t = 0; t < N_i.size(); t++){
        for(int s = t+1; s < N_i.size(); s++){
            if(s==t){
                continue;
            }
            float updated_wt = (L.coeff(i, N_i[t])*L.coeff(i, N_i[t])/L.coeff(i, i));
            D_ts.insert(N_i[t], edge_idx) = 1.0 * std::sqrt(updated_wt);
            D_ts.insert(N_i[s], edge_idx) = -1.0 * std::sqrt(updated_wt);
            edge_idx += 1;
            D_ts.insert(N_i[s], edge_idx) = 1.0 * std::sqrt(updated_wt);
            D_ts.insert(N_i[t], edge_idx) = -1.0 * std::sqrt(updated_wt);
            edge_idx += 1;
        }
    }
    D_ts.makeCompressed();
    return (D_ts * D_ts.transpose())/2;
}

Eigen::MatrixXf WeightedCliqueSampler::sampleClique(Eigen::SparseMatrix<float>& L, int i){
    std::vector<int> N_i;
    std::vector<float> N_w;
    for(int k = 0; k < L.cols(); k++){
        if(i!=k && std::abs(L.coeff(i, k)) >= 1e-8 ){
            N_i.push_back(k);
            N_w.push_back(std::abs(L.coeff(i, k)));
        }
    }
    // std::cout << "Node " << i << " has neighbour count = "<< N_i.size() << std::endl;
    std::mt19937_64 rand_generator;
    std::discrete_distribution<int> p_distribution(N_w.begin(), N_w.end());

    std::uniform_int_distribution<int> u_distribution(0, N_i.size()-1);

    Eigen::SparseMatrix<float> D_ts(L.rows(), L.cols());
    D_ts.reserve(Eigen::VectorXi::Constant(L.cols(), 2));
    int edge_idx = 0;
    while(edge_idx < N_i.size()-1){
        int n1 = N_i[(int)p_distribution(rand_generator)];
        int n2 = N_i[(int)u_distribution(rand_generator)];
        float w_i_n1 = std::abs(L.coeff(i, n1));
        float w_i_n2 = std::abs(L.coeff(i, n2));
        // std::cout << "N1 = " << n1 << " WN1 = " << w_i_n1 << " N2 = " << n2 << " WN2 = " << w_i_n2 << std::endl;
        if(n1==n2 || w_i_n1 < 1e-8 || w_i_n2 < 1e-8){
            continue;
        }
        float scaled_wt = (w_i_n1*w_i_n2)/(w_i_n1 + w_i_n2);
        D_ts.insert(n1, edge_idx) = 1.0 * std::sqrt(scaled_wt);
        D_ts.insert(n2, edge_idx) = -1.0 * std::sqrt(scaled_wt);
        edge_idx += 1;
    }
    return D_ts * D_ts.transpose();
}
