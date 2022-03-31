#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/Eigen/SparseCholesky"
#include "third_party/eigen3/Eigen/Core"
#include <iostream>
#include "cg.h"
#include "types.h"

ConjugateGradient::ConjugateGradient(Eigen::MatrixXf M, Eigen::VectorXf b){
    _M = M;
    _b = b;
}

Eigen::VectorXf ConjugateGradient::solve(float tolerance, int max_iters){
    Eigen::VectorXf x = Eigen::VectorXf::Zero(_b.size());
    Eigen::VectorXf r = _b;
    Eigen::VectorXf p = r;
    for(int i = 0; i < max_iters; i++){
        // calculate the step length
        float den = float(p.transpose() * _M * p);
        if(den <= 1e-14 || std::isinf(den)){
            std::cout << "CG: Early stopping due to large or small den for alpha" << std::endl;
            std::cout << "CG: Iteration count = " << i+1 << std::endl;
            break;
        }
        float alpha = float(r.transpose() * r) / float(p.transpose() * _M * p);
        // std::cout << "Alpha = " << alpha << std::endl;
        // approximate the solution by moving in the search direction
        x = x + alpha * p;
        // calculate the residual
        Eigen::VectorXf r_n = r - alpha * _M * p;
        // improvement in this direction
        float beta = float(r_n.transpose() * r_n)/float(r.transpose() * r);
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

PConjugateGradient::PConjugateGradient(Eigen::SparseMatrix<float>* M, Eigen::VectorXf* b){
    _M = *M;
    _b = *b;
}

Eigen::VectorXf PConjugateGradient::solve(float tolerance, int max_iters){
    // std::cout << "tolerance = " << tolerance << " max_iters = " << max_iters << std::endl;
    int stag_test = 5;

    double al;
    double n = _M.cols();
    double nb = _b.norm();
    if(nb == 0){
        return Eigen::VectorXf::Zero(_b.size());
    }

    Eigen::VectorXf x = Eigen::VectorXf::Zero(n);
    Eigen::VectorXf best_x = Eigen::VectorXf::Zero(n);
    double bestnr = 1.0;

    Eigen::VectorXf r = _b;
    Eigen::VectorXf z = this->applyPreconditioner(r);
    // Eigen::VectorXf z = r;
    Eigen::VectorXf p = z;

    double rho = r.dot(z);
    double best_rho = rho;
    double stag_count = 0;

    int iter_count = 0;
    while(iter_count < max_iters){
        iter_count += 1;
        // std::cout << "CG ITER COUNT = " << iter_count << std::endl;
        Eigen::VectorXf q = _M * p;
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
        // z = r;
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
    return best_x;
}

void PConjugateGradient::setPreconditioner(LDLi* ldli){
    _ldli = ldli;
}

Eigen::VectorXf PConjugateGradient::applyPreconditioner(Eigen::VectorXf b){

    Eigen::VectorXf y = b;
    // Forward pass
    // std::cout << "col len = " << _ldli->col.size() << " colptr len = " << _ldli->colptr.size();
    // std::cout << " fval len = " << _ldli->fval.size() << " rowval len = " << _ldli->rowval.size() << " d len = " << _ldli->d.size() << std::endl;
    // std::cout << "FORWARD PASS" << std::endl;
    for(int ii = 0; ii < _ldli->col.size(); ii++){
        float i = _ldli->col.at(ii);
        // std::cout << "i = " << i << std::endl;
        float j0 = _ldli->colptr.at(ii);
        // std::cout << "j0 = " << j0 << std::endl;
        float j1 = _ldli->colptr.at(ii+1)-1;
        // std::cout << "j1 = " << j1 << std::endl;
        float yi = y.coeff(i);

        for(int jj = j0; jj < j1; jj++){
            float j = _ldli->rowval.at(jj);
            // std::cout << "j = " << j << std::endl;
            // std::cout << "fval = " << _ldli->fval.at(jj) << std::endl;
            y.coeffRef(j) += _ldli->fval.at(jj)*yi;
            yi *= (1-_ldli->fval.at(jj));
        }
        float j = _ldli->rowval.at(j1);
        // std::cout << "j = " << j << std::endl;
        y.coeffRef(j) += yi;
        y.coeffRef(i) = yi;
    }

    // Diagonal pass
    // std::cout << "DIAGONAL PASS" << std::endl;
    for(int i = 0; i < _ldli->d.size(); i++){
        if(_ldli->d.at(i) != 0){
            y.coeffRef(i) /= _ldli->d.at(i);
        }
    }

    // Backward pass
    // std::cout << "BACKWARD PASS" << std::endl;
    for(int ii = _ldli->col.size()-1; ii > -1 ; ii--){
        float i = _ldli->col.at(ii);

        float j0 = _ldli->colptr.at(ii);
        float j1 = _ldli->colptr.at(ii+1)-1;

        float j = _ldli->rowval.at(j1);
        float yi = y.coeff(i);
        yi += y.coeff(j);

        for(int jj = j1-1; jj > j0-1; jj--){
            float j = _ldli->rowval.at(jj);
            yi = ( 1 - _ldli->fval.at(jj) )*yi + _ldli->fval.at(jj)*y.coeff(j);
        }
        y.coeffRef(i) = yi;
    }

    float y_mean = y.mean();
    y = y - Eigen::VectorXf::Ones(y.size())*y_mean;
    return y;
}

