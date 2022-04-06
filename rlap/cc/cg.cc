#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include <iostream>
#include <chrono>
#include "cg.h"
#include "types.h"

ConjugateGradient::ConjugateGradient(Eigen::MatrixXd M, Eigen::VectorXd b){
    _M = M;
    _b = b;
}

Eigen::VectorXd ConjugateGradient::solve(double tolerance, int max_iters){
    Eigen::VectorXd x = Eigen::VectorXd::Zero(_b.size());
    Eigen::VectorXd r = _b;
    Eigen::VectorXd p = r;
    for(int i = 0; i < max_iters; i++){
        // calculate the step length
        double den = double(p.transpose() * _M * p);
        if(den <= 1e-14 || std::isinf(den)){
            std::cout << "CG: Early stopping due to large or small den for alpha" << std::endl;
            std::cout << "CG: Iteration count = " << i+1 << std::endl;
            break;
        }
        double alpha = double(r.transpose() * r) / double(p.transpose() * _M * p);
        // std::cout << "Alpha = " << alpha << std::endl;
        // approximate the solution by moving in the search direction
        x = x + alpha * p;
        // calculate the residual
        Eigen::VectorXd r_n = r - alpha * _M * p;
        // improvement in this direction
        double beta = double(r_n.transpose() * r_n)/double(r.transpose() * r);
        // std::cout << "Beta = " << beta << std::endl;
        // update the search direction
        p = r_n + beta * p;
        // assign the new value to the residual
        r = r_n;
        // early stopping if we are within the tolerance level of the solution
        if(r.norm() <= tolerance){
            std::cout << "CG: Early stopping due to tolerance limit" << std::endl;
            std::cout << "CG: Iteration count = " << i+1 << std::endl;
            break;
        }
    }
    return x;
}

PConjugateGradient::PConjugateGradient(Eigen::SparseMatrix<double>* M, Eigen::VectorXd* b){
    _M = *M;
    _b = *b;
}

Eigen::VectorXd PConjugateGradient::solve(double tolerance, int max_iters){
    // std::cout << "tolerance = " << tolerance << " max_iters = " << max_iters << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    int stag_test = 5;

    double al;
    double n = _M.cols();
    double nb = _b.norm();
    if(nb == 0){
        return Eigen::VectorXd::Zero(_b.size());
    }

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd best_x = Eigen::VectorXd::Zero(n);
    double bestnr = 1.0;

    Eigen::VectorXd r = _b;
    Eigen::VectorXd z = this->applyPreconditioner(r);
    Eigen::VectorXd p = z;

    double rho = r.dot(z);
    double best_rho = rho;
    double stag_count = 0;

    int iter_count = 0;
    while(iter_count < max_iters){
        iter_count += 1;
        // std::cout << "CG ITER COUNT = " << iter_count << std::endl;
        Eigen::VectorXd q = _M * p;
        double pq = p.dot(q);
        if(pq <= 1e-16 || std::isinf(pq)){
            std::cout << "CG: Early stopping due to large or small pq" << std::endl;
            break;
        }

        al = double(rho)/pq;

        if(al*p.norm() < 1e-16*x.norm()){
            std::cout << "CG: Early stopping due to stagnation" << std::endl;
            break;
        }

        // std::cout << "pq = " << pq <<  " al = " << al << " rho = " << rho << std::endl;

        x = x + al*p;
        r = r - al*q;

        double nr = double(r.norm())/nb;

        if(nr < bestnr){
            bestnr = nr;
            best_x = x;
        }
        // std::cout << "nr = " << nr << " bestnr = " << bestnr << std::endl;
        if(nr < tolerance){
            break;
        }

        z = this->applyPreconditioner(r);
        double oldrho = rho;
        rho = z.dot(r);
        if(rho < best_rho*(1-1/stag_test)){
            best_rho = rho;
            stag_count = 0;
        }
        else if(stag_test > 0 && best_rho > (1-1/stag_test)*rho){
            stag_count += 1;
            if (stag_count > stag_test){
                std::cout << "CG: Early stopping due to stagnation test. stag_test = " << stag_test << std::endl;
                break;
            }
        }

        if(rho <= 1e-16 || std::isinf(rho)){
            std::cout << "CG: Early stopping due to large or small rho " << rho << std::endl;
            break;
        }

        double beta = double(rho)/oldrho;
        if(beta <= 1e-16 || std::isinf(beta)){
            std::cout << "CG: Early stopping due to large or small beta" << std::endl;
            break;
        }

        p = z + beta*p;
        // std::cout<< " oldrho = " << oldrho << " rho = " << rho << " beta= " << beta << std::endl;
    }
    std::cout << "CG stopped after Iteration count = " << iter_count << " with error = " << r.norm()/nb << std::endl;
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
    return best_x;
}

void PConjugateGradient::setPreconditioner(LDLi* ldli){
    _ldli = ldli;
}

Eigen::VectorXd PConjugateGradient::applyPreconditioner(Eigen::VectorXd b){

    Eigen::VectorXd y = b;
    // Forward pass
    // std::cout << "col len = " << _ldli->col.size() << " colptr len = " << _ldli->colptr.size();
    // std::cout << " fval len = " << _ldli->fval.size() << " rowval len = " << _ldli->rowval.size() << " d len = " << _ldli->d.size() << std::endl;
    // std::cout << "FORWARD PASS" << std::endl;
    for(int ii = 0; ii < _ldli->col.size(); ii++){
        double i = _ldli->col.at(ii);
        // std::cout << "i = " << i << std::endl;
        double j0 = _ldli->colptr.at(ii);
        // std::cout << "j0 = " << j0 << std::endl;
        double j1 = _ldli->colptr.at(ii+1)-1;
        // std::cout << "j1 = " << j1 << std::endl;
        double yi = y.coeff(i);

        for(int jj = j0; jj < j1; jj++){
            double j = _ldli->rowval.at(jj);
            // std::cout << "j = " << j << std::endl;
            // std::cout << "fval = " << _ldli->fval.at(jj) << std::endl;
            y.coeffRef(j) += _ldli->fval.at(jj)*yi;
            yi *= (1-_ldli->fval.at(jj));
        }
        if(j1 >= 0){
            double j = _ldli->rowval.at(j1);
            // std::cout << "j = " << j << std::endl;
            y.coeffRef(j) += yi;
            y.coeffRef(i) = yi;
        }
    }
    // std::cout << "FP Norm of y = " << y.norm() << std::endl;
    // std::cout << "FP Mean of y = " << y.mean() << std::endl;

    // Diagonal pass
    // std::cout << "DIAGONAL PASS" << std::endl;
    for(int i = 0; i < _ldli->d.size(); i++){
        if(_ldli->d.at(i) != 0){
            y.coeffRef(i) /= _ldli->d.at(i);
        }
    }

    // std::cout << "D Norm of y = " << y.norm() << std::endl;
    // std::cout << "D Mean of y = " << y.mean() << std::endl;

    // Backward pass
    // std::cout << "BACKWARD PASS" << std::endl;
    for(int ii = _ldli->col.size()-1; ii > -1 ; ii--){
        double i = _ldli->col.at(ii);

        double j0 = _ldli->colptr.at(ii);
        double j1 = _ldli->colptr.at(ii+1)-1;
        if(j1 < 0){
            continue;
        }
        double j = _ldli->rowval.at(j1);
        double yi = y.coeff(i);
        yi += y.coeff(j);

        for(int jj = j1-1; jj > j0-1; jj--){
            double j = _ldli->rowval.at(jj);
            yi = ( 1 - _ldli->fval.at(jj) )*yi + _ldli->fval.at(jj)*y.coeff(j);
        }
        y.coeffRef(i) = yi;
    }

    // std::cout << "BP Norm of y = " << y.norm() << std::endl;
    // std::cout << "BP Mean of y = " << y.mean() << std::endl;

    double y_mean = y.mean();
    y = y - Eigen::VectorXd::Ones(y.size())*y_mean;
    return y;
}

