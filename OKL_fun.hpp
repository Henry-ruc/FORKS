#ifndef OKL
#define OKL
#include <iostream>
#include<cmath>
#include <sys/timeb.h>
#include <ctime>
#include <chrono>
#include <random>
#include<math.h>
#include<cstdio>
#include<complex>
#include<iostream>
#include<algorithm>
#include<vector>
#include<fstream>
#include <iostream>
#include<sstream>
#include<Eigen/Core>
#include<Eigen/SVD>
#include <Eigen/Dense>
#include "murmur3_hash.hpp"
#include "Murmurhash.h"
#define PI M_PI
using namespace std;
using namespace chrono;  
using namespace Eigen;
using namespace Eigen::internal;  
using namespace Eigen::Architecture;  
using namespace std;
namespace OKL{

class SJLT
{
private:
    /* data */
public:
    SJLT(int n,int s,int d=1);

    int n;//row
    int s;//col
    int d;//block

    MatrixXf sketch_matrix;

    std::vector<std::vector<int> > seeds1;
    std::vector<std::vector<int> > seeds2;

    MatrixXf generation();


    ~SJLT();
};

SJLT::SJLT(int n,int s,int d)
{
    this->n = n;
    this->d = d;
    this->s = s;

    for (unsigned int i = 0; i < d; i++) 
    {
        std::vector<int> seeds1_item;
        std::vector<int> seeds2_item;
        for (int i = 0; i < n; i++)
        {
            seeds1_item.push_back(rand());
            seeds2_item.push_back(rand());
        }
        seeds1.push_back(seeds1_item);
        seeds2.push_back(seeds2_item);
    }
}

SJLT::~SJLT()
{
}

MatrixXf SJLT::generation()
{
    sketch_matrix.resize(n,s);
    MatrixXf sketch_matrix_temp;
    sketch_matrix_temp.resize(n,s/d);
    for (int i = 0; i < d; i++)
    {
        MatrixXf SJLT_item = MatrixXf::Zero(n,s/d);
        for (int p = 0; p < n; p++)
        {
            int q = murmurhash(&p,seeds1[i][p])%(s/d);
            SJLT_item(p,q) = (int(murmurhash(&p,seeds2[i][p])%2)*2-1)/sqrt(d);
        }
        if (i==0)
        {
            sketch_matrix_temp = SJLT_item;
        }
        else
        {
            sketch_matrix.resize(n,(i+1)*s/d);
            sketch_matrix << sketch_matrix_temp,SJLT_item;
            sketch_matrix_temp.resize(n,(i+1)*s/d);
            sketch_matrix_temp = sketch_matrix;
        } 
    }
    if (d==1)
    {
        sketch_matrix = sketch_matrix_temp;
    }
    return sketch_matrix;
}

class SkeGD
{
private:
    /* data */
public:
    SkeGD(int sp,int sm,float eta,int k,int B,int rou,float sigma,int d):sketch(B,sp,d)
    {
        this->sp = sp;
        this->sm = sm;
        this->eta=eta;
        this-> k = k;
        this->B = B;
        this->rou = rou;
        this->sigma = sigma;
        this->d = d;
        wrong = 0;
    };
    ~SkeGD();

    int sp;//sketch size
    int sm;//sample size
    float eta;//stepsize
    int k;//SVD rank
    int B;//budget
    int rou;//updae cycle
    int d;
    float sigma;

    MatrixXf SV_1;
    MatrixXf mat_phi_pm;
    MatrixXf mat_phi_pp;
    MatrixXf mat_Q;
    MatrixXf weight;
    SJLT sketch;
    std::vector<MatrixXf> colsampleMat;
    std::vector<float> alpha;
    int wrong;

    void Initialization_sketch(MatrixXf x,float y);
    void Update_sketch(MatrixXf x,float y,int flag=1);
    void SkeGD_learning(MatrixXf dataX,MatrixXf dataY);

    int fun_sign(double x); 
    MatrixXf compute_K_SkeGD(MatrixXf X,MatrixXf Y,float sigma);//compute K
    std::vector<MatrixXf> colsampling(MatrixXf mat,int col);
    Eigen::MatrixXf pinv(Eigen::MatrixXf  A);

    std::vector<float> loss;



};

SkeGD::~SkeGD()
{
}

MatrixXf SkeGD::compute_K_SkeGD(MatrixXf X,MatrixXf Y,float sigma)
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));//Gaussian kernel
            //K(i,j) = (mat_id.row(i)*mat_SV.row(j).transpose())(0,0);//linear
        }
    }
    return K;
}

int SkeGD::fun_sign(double x)
{
    if(x>=0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

std::vector<MatrixXf> SkeGD::colsampling(MatrixXf mat,int col)
{
    std::vector<MatrixXf> output;
    MatrixXf mat_I = MatrixXf::Identity(mat.cols(),mat.cols());
    MatrixXf mat_Sm = MatrixXf::Zero(mat.cols(),col);
    
    std::vector<int> index;
    for (int i = 0; i < mat.cols(); i++)
    {
        index.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(index.begin(), index.end(), g);

    for (int i = 0; i < col; i++)
    {
        mat_Sm.col(i) = mat_I.col(index[i]);
    }

    output.push_back(mat_Sm);
    output.push_back(mat*mat_Sm);
    return output;
}

Eigen::MatrixXf SkeGD::pinv(Eigen::MatrixXf  A)
{
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double  pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = min(row,col);
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(col,row);
    Eigen::MatrixXf singularValues_inv = svd.singularValues();
    Eigen::MatrixXf singularValues_inv_mat = Eigen::MatrixXf::Zero(col, row);
    for (long i = 0; i<k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i) 
    {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X=(svd.matrixV())*(singularValues_inv_mat)*(svd.matrixU().transpose());
 
    return X;
}

void SkeGD::Initialization_sketch(MatrixXf x,float y)
{
    MatrixXf K_B = compute_K_SkeGD(SV_1,SV_1,sigma);
    sketch.generation();

    MatrixXf mat_C_p = K_B*sketch.sketch_matrix;
    
    colsampleMat = colsampling(K_B,sm);
    MatrixXf mat_C_m = colsampleMat[1];
    mat_phi_pm = sketch.sketch_matrix.transpose()*mat_C_m;
    mat_phi_pp = sketch.sketch_matrix.transpose()*mat_C_p;

    //SVD
    JacobiSVD<MatrixXf> svd(mat_phi_pp,ComputeThinU | ComputeThinV);
    MatrixXf V = svd.matrixV(), U = svd.matrixU();
    MatrixXf Vk = V.block(0,0,V.rows(),k);
    MatrixXf Uk = U.block(0,0,U.rows(),k);
    MatrixXf S_sqrt = MatrixXf::Zero(k,k);
    for (int j = 0; j < k; j++)
    {
        S_sqrt(j,j) = sqrt(svd.singularValues()(j,0));
    }
    mat_Q = pinv(mat_phi_pm)*Vk*S_sqrt;

    MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma);
    MatrixXf feature_mapping = (mat_psi*colsampleMat[0]*mat_Q).transpose();

    MatrixXf mat_alpha = Map<MatrixXf, Eigen::Unaligned>(alpha.data(), 1,alpha.size());//To matrix
    weight = mat_alpha*(mat_psi.transpose())*pinv(feature_mapping);//1*k
    MatrixXf gradient = eta*y*(feature_mapping.transpose());
    weight = weight+gradient;//1*k
}

void SkeGD::Update_sketch(MatrixXf x,float y,int flag)
{
    if(flag==1)
    {
        MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma).transpose();//B*1
        SJLT ske_sub(1,sp,d);
        ske_sub.generation();
        MatrixXf mat_R_pm = (ske_sub.sketch_matrix.transpose())*(mat_psi.transpose()*colsampleMat[0]);
        mat_phi_pm = mat_phi_pm+mat_R_pm;//update
        MatrixXf mat_R_pp = ske_sub.sketch_matrix.transpose()*(mat_psi.transpose()*sketch.sketch_matrix);
        MatrixXf xi = compute_K_SkeGD(x,x,sigma);
        MatrixXf mat_T_pp = xi(0,0)*ske_sub.sketch_matrix.transpose()*ske_sub.sketch_matrix;
        mat_phi_pp = mat_phi_pp+mat_R_pp+mat_R_pp.transpose()+mat_T_pp;
        //SVD
        JacobiSVD<MatrixXf> svd(mat_phi_pp,ComputeThinU | ComputeThinV);
        MatrixXf V = svd.matrixV(), U = svd.matrixU();
        MatrixXf Vk = V.block(0,0,V.rows(),k);
        MatrixXf Uk = U.block(0,0,U.rows(),k);
        MatrixXf S_sqrt = MatrixXf::Zero(k,k);
        for (int j = 0; j < k; j++)
        {
            S_sqrt(j,j) = sqrt(svd.singularValues()(j,0));
        }
        mat_Q = pinv(mat_phi_pm)*Vk*S_sqrt;
        //Feature mapping
        MatrixXf feature_mapping = (mat_psi.transpose()*colsampleMat[0]*mat_Q).transpose();//k*1
        //update weight
        // weight = (weight*feature_mapping)(0,0)*pinv(feature_mapping);

        //Compute gradient
        MatrixXf gradient = eta*y*(feature_mapping.transpose());//hinge loss
        // float fx = (weight*feature_mapping)(0,0);//squared hinge loss
        // MatrixXf gradient = 2*eta*y*(1-y*fx)*(feature_mapping.transpose());
        weight = weight+gradient;//1*k
    }
    else
    {
        MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma).transpose();
        
        MatrixXf feature_mapping = (mat_psi.transpose()*colsampleMat[0]*mat_Q).transpose();

        
        MatrixXf gradient = eta*y*(feature_mapping.transpose());//hinge loss
        // float fx = (weight*feature_mapping)(0,0);//squared hinge loss
        // MatrixXf gradient = 2*eta*y*(1-y*fx)*(feature_mapping.transpose());
        weight = weight+gradient;//1*k
    }

}



void SkeGD::SkeGD_learning(MatrixXf dataX,MatrixXf dataY)
{
    int wheather_init = 0;
    int T0 = 0;
    for (int t = 0; t < dataX.rows(); t++)
    {
        MatrixXf xt = dataX.row(t);
        float yt = dataY(0,t);
        float f_t;
        
        // compute f_t(x_t)
        if(alpha.size()==0)
        {
            f_t = 0;
        }
        else
        {
            if (wheather_init == 0)
            {
                MatrixXf k_t = compute_K_SkeGD(SV_1,xt,sigma);
                MatrixXf mat_alpha = Map<MatrixXf, Eigen::Unaligned>(alpha.data(), 1,alpha.size());
                f_t = (mat_alpha*k_t)(0,0);
            }
            else
            {
                MatrixXf k_b = compute_K_SkeGD(xt,SV_1,sigma)*colsampleMat[0];//1*sm
                MatrixXf phi_t = (k_b*mat_Q).transpose();//k*1
                // std::cout<<"weight:"<<weight<<endl;
                // std::cout<<"phi:"<<phi_t.transpose()<<endl;
                f_t = (weight*phi_t)(0,0);
            }
        }
        float hat_y_t = fun_sign(f_t);
        //std::cout<<f_t<<" "<<yt<<endl;

        //count the number of errors
        if(abs(hat_y_t-yt)>0.0001 and wheather_init==1)
        {
            wrong++;
        }

        if(1-yt*f_t>0)
        {
            loss.push_back(yt*f_t);
        }
        else
        {
            loss.push_back(0);
        }

        if(yt*f_t<1)
        {
            T0++;
            if (SV_1.rows()<B)
            {
                alpha.push_back(eta*yt);
                if (SV_1.rows()==0)
                {
                    SV_1 = MatrixXf::Zero(1,xt.cols());
                    SV_1.row(0) = xt.row(0);
                }
                else
                {
                    SV_1.conservativeResize(SV_1.rows()+1, SV_1.cols());
                    SV_1.row(SV_1.rows()-1) = xt.row(0);
                }
            }
            else
            {
                if(SV_1.rows() == B and wheather_init == 0)
                {
                    Initialization_sketch(xt,yt);
                    wheather_init = 1;
                }
                else
                {
                    if (T0%rou == 1)//update
                    {
                        Update_sketch(xt,yt);
                    }
                    else
                    {
                        Update_sketch(xt,yt,0);
                    }
                }
            }
        }
    }
    //std::cout<<"T0:"<<T0<<endl; 
};

class ISVD_lowrank
{
private:
    /* data */
public:
    ISVD_lowrank(int k);
    ~ISVD_lowrank();

    MatrixXf U;//t*k
    MatrixXf S;//k*k
    MatrixXf V;//t*k
    int k;

    void update(MatrixXf A,MatrixXf B);//a:t*3 b:t*3
    void init(MatrixXf X);//X:t*t;
};

ISVD_lowrank::ISVD_lowrank(int k)
{
    this->k = k;
}

ISVD_lowrank::~ISVD_lowrank()
{
}

void ISVD_lowrank::update(MatrixXf A,MatrixXf B)
{
    MatrixXf I = MatrixXf::Identity(U.rows(),U.rows());
    MatrixXf p = (I-U*U.transpose())*A;//t*3
    MatrixXf q = (I-V*V.transpose())*B;//t*3

    CompleteOrthogonalDecomposition<MatrixXf> cod;
    cod.compute(p);
    MatrixXf P = cod.matrixQ();//t*3
    P.conservativeResize(p.rows(),3);
    //P = P.block(0,0,p.rows(),3);
    MatrixXf RA = P.transpose()*A;//3*3
    RA.conservativeResize(3,3);
    // RA = RA.block(0,0,RA.cols(),RA.cols());

    CompleteOrthogonalDecomposition<MatrixXf> cod2;
    cod2.compute(q);
    MatrixXf Q = cod.matrixQ();//t*3
    //Q = Q.block(0,0,q.rows(),3);
    Q.conservativeResize(q.rows(),3);
    MatrixXf RB = Q.transpose()*B;//3*3
    //RB = RB.block(0,0,RB.cols(),RB.cols());
    RB.conservativeResize(3,3);

    MatrixXf UP(U.rows(),U.cols()+P.cols());//t*(k+3)
    MatrixXf VQ(V.rows(),V.cols()+Q.cols());//t*(k+3)
    UP<<U,P;
    VQ<<V,Q;
    //Compute H
    int rankAB = A.cols();
    MatrixXf S11 = S;
    MatrixXf zero_left = MatrixXf::Zero(S11.rows(),rankAB);
    MatrixXf S10(k,k+rankAB);
    S10<<S11,zero_left;
    MatrixXf zero_down = MatrixXf::Zero(rankAB,k+rankAB);
    MatrixXf S1(k+rankAB,k+rankAB);//(k+3)*(k+3)
    S1<<S10,
    zero_down;
    MatrixXf UARA(k+rankAB,A.cols());//(k+3)*3
    UARA<<U.transpose()*A,
    RA;
    MatrixXf VBRB(k+rankAB,B.cols());//(k+3)*3
    VBRB<<V.transpose()*B,
    RB;
    MatrixXf H = S1+UARA*VBRB.transpose();//(k+3)*(k+3)
    JacobiSVD<MatrixXf> svd(H,ComputeThinU | ComputeThinV);
    MatrixXf H_V = svd.matrixV();//(k+3)*(k+3)
    MatrixXf H_U = svd.matrixU();//(k+3)*(k+3)
    H_V.conservativeResize(k+3,k);
    H_U.conservativeResize(k+3,k);
    MatrixXf H_S = MatrixXf::Zero(k,k);
    for (int i = 0; i < k; i++)
    {
        H_S(i,i) = svd.singularValues()(i,0);
    }
    U = UP*H_U;//t*k
    V = VQ*H_V;//t*k
    S = H_S;//k*k
}

void ISVD_lowrank::init(MatrixXf X)
{
    JacobiSVD<MatrixXf> svd(X,ComputeThinU | ComputeThinV);
    V = svd.matrixV();
    U = svd.matrixU();
    V.conservativeResize(V.rows(),k);
    U.conservativeResize(U.rows(),k);
    // V = V.block(0,0,V.rows(),k);//t*k
    // U = U.block(0,0,U.rows(),k);//t*k
    S = MatrixXf::Zero(k,k);//k*k
    for (int j = 0; j < k; j++)
    {
        S(j,j) = svd.singularValues()(j,0);
    }
}


class SkeGD2nd
{
private:
    /* data */
public:
    SkeGD2nd(int sp,int sm,float eta,int k,int B,int rou,float sigma,int d,float regularizer):sketch(B,sp,d),ISVD_for_SkeGD(k)
    {
        this->sp = sp;
        this->sm = sm;
        this->eta=eta;
        this-> k = k;
        this->B = B;
        this->rou = rou;
        this->sigma = sigma;
        this->d = d;
        this->regularizer = regularizer;
        sigma_t = 1;
        wrong = 0;
        batch_wrong = 0;
    };
    ~SkeGD2nd();

    int sp;//sketch size
    int sm;//sample size
    float eta;//stepsize
    int k;//SVD rank
    int B;//budget
    int rou;//updae cycle
    int d;
    float sigma;
    float sigma_t;
    float regularizer;

    MatrixXf SV_1;
    MatrixXf mat_phi_pm;
    MatrixXf mat_phi_pp;
    MatrixXf mat_Q;
    MatrixXf weight;
    MatrixXf v_t;
    MatrixXf mat_A_t_inv;
    ISVD_lowrank ISVD_for_SkeGD;
    SJLT sketch;
    std::vector<MatrixXf> colsampleMat;
    std::vector<float> alpha; 
    int wrong;
    int batch_wrong;

    void Initialization_sketch(MatrixXf x,float y);
    void Update_sketch(MatrixXf x,float y,int flag=1);
    void SkeGD_learning(MatrixXf dataX,MatrixXf dataY);


    int fun_sign(double x); 
    int fun_sign2(float x);
    MatrixXf compute_K_SkeGD(MatrixXf X,MatrixXf Y,float sigma);
    std::vector<MatrixXf> colsampling(MatrixXf mat,int col);
    Eigen::MatrixXf pinv(Eigen::MatrixXf  A);
    float fun_h(float x,float C=1);

    std::vector<float> loss;

};

SkeGD2nd::~SkeGD2nd()
{
}

MatrixXf SkeGD2nd::compute_K_SkeGD(MatrixXf X,MatrixXf Y,float sigma)
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));
            //K(i,j) = (mat_id.row(i)*mat_SV.row(j).transpose())(0,0);
        }
    }
    return K;
}

int SkeGD2nd::fun_sign(double x)
{
    if(x>=0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int SkeGD2nd::fun_sign2(float x)
{
    if(abs(x-0)<0.0001)
    {
        return 0;
    }
    else if (x>0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

float SkeGD2nd::fun_h(float x,float C)
{
    float anw = 0;
    if((abs(x)-C)>anw)
    {
        anw = abs(x)-C;
    }
    return fun_sign2(x)*anw;
}

std::vector<MatrixXf> SkeGD2nd::colsampling(MatrixXf mat,int col)
{
    std::vector<MatrixXf> output;
    MatrixXf mat_I = MatrixXf::Identity(mat.cols(),mat.cols());
    MatrixXf mat_Sm = MatrixXf::Zero(mat.cols(),col);
    
    std::vector<int> index;
    for (int i = 0; i < mat.cols(); i++)
    {
        index.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(index.begin(), index.end(), g);

    for (int i = 0; i < col; i++)
    {
        mat_Sm.col(i) = mat_I.col(index[i]);
    }

    output.push_back(mat_Sm);
    output.push_back(mat*mat_Sm);
    return output;
}

Eigen::MatrixXf SkeGD2nd::pinv(Eigen::MatrixXf  A)
{
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double  pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = min(row,col);
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(col,row);
    Eigen::MatrixXf singularValues_inv = svd.singularValues();
    Eigen::MatrixXf singularValues_inv_mat = Eigen::MatrixXf::Zero(col, row);
    for (long i = 0; i<k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i) 
    {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X=(svd.matrixV())*(singularValues_inv_mat)*(svd.matrixU().transpose());
 
    return X;
}

void SkeGD2nd::Initialization_sketch(MatrixXf x,float y)
{
    MatrixXf K_B = compute_K_SkeGD(SV_1,SV_1,sigma); 
    sketch.generation();

    MatrixXf mat_C_p = K_B*sketch.sketch_matrix;

    colsampleMat = colsampling(K_B,sm);
    MatrixXf mat_C_m = colsampleMat[1];
    mat_phi_pm = sketch.sketch_matrix.transpose()*mat_C_m;
    mat_phi_pp = sketch.sketch_matrix.transpose()*mat_C_p;

    //ISVD
    ISVD_for_SkeGD.init(mat_phi_pp);
    MatrixXf Vk = ISVD_for_SkeGD.V;
    MatrixXf S_sqrt = MatrixXf::Zero(k,k);
    for (int j = 0; j < k; j++)
    {
        S_sqrt(j,j) = sqrt(ISVD_for_SkeGD.S(j,j));
    }

    mat_Q = pinv(mat_phi_pm)*Vk*S_sqrt;

    MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma);
    MatrixXf feature_mapping = (mat_psi*colsampleMat[0]*mat_Q).transpose();//k*1
    //weight
    MatrixXf mat_alpha = Map<MatrixXf, Eigen::Unaligned>(alpha.data(), 1,alpha.size());
    weight = mat_alpha*(mat_psi.transpose())*pinv(feature_mapping);//1*k
    mat_A_t_inv = (1.0/regularizer)*MatrixXf::Identity(k,k);//Init A
    MatrixXf gradient = eta*y*(feature_mapping);//k*1
    v_t = weight.transpose() + mat_A_t_inv*gradient;//k*1
    float coef = fun_h((feature_mapping.transpose()*v_t)(0,0))/(feature_mapping.transpose()*(mat_A_t_inv)*feature_mapping)(0,0);
    weight = (v_t-coef*(mat_A_t_inv)*feature_mapping).transpose();//1*k

    MatrixXf u = gradient/(sigma_t);
    MatrixXf A_inv_dot_u = (mat_A_t_inv*u).transpose();//1*k
    mat_A_t_inv -= (mat_A_t_inv*u*u.transpose()*mat_A_t_inv)/(1+(A_inv_dot_u*u)(0,0));

}

void SkeGD2nd::Update_sketch(MatrixXf x,float y,int flag)
{
    if(flag==1)
    {
        MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma).transpose();//B*1
        SJLT ske_sub(1,sp,d);
        ske_sub.generation();
        MatrixXf mat_R_pm = (ske_sub.sketch_matrix.transpose())*(mat_psi.transpose()*colsampleMat[0]);
        mat_phi_pm = mat_phi_pm+mat_R_pm;//update
        MatrixXf mat_R_pp = ske_sub.sketch_matrix.transpose()*(mat_psi.transpose()*sketch.sketch_matrix);
        MatrixXf xi = compute_K_SkeGD(x,x,sigma);
        MatrixXf mat_T_pp = xi(0,0)*ske_sub.sketch_matrix.transpose()*ske_sub.sketch_matrix;
        mat_phi_pp = mat_phi_pp+mat_R_pp+mat_R_pp.transpose()+mat_T_pp;//update

        int t = ske_sub.sketch_matrix.transpose().rows();
        MatrixXf deltaA(t,3);
        MatrixXf deltaB(t,3);
        deltaA.col(0) = ske_sub.sketch_matrix.transpose();
        deltaA.col(1) = (mat_psi.transpose()*sketch.sketch_matrix).transpose();
        deltaA.col(2) = xi(0,0)*ske_sub.sketch_matrix.transpose();
        deltaB.col(0) = (mat_psi.transpose()*sketch.sketch_matrix).transpose();
        deltaB.col(1) = ske_sub.sketch_matrix.transpose();
        deltaB.col(2) = ske_sub.sketch_matrix.transpose();
        ISVD_for_SkeGD.update(deltaA,deltaB);

        MatrixXf Vk = ISVD_for_SkeGD.V;
        MatrixXf S_sqrt = MatrixXf::Zero(k,k);
        for (int j = 0; j < k; j++)
        {
            S_sqrt(j,j) = sqrt(ISVD_for_SkeGD.S(j,j));
        }

        mat_Q = pinv(mat_phi_pm)*Vk*S_sqrt;
        
        MatrixXf feature_mapping = (mat_psi.transpose()*colsampleMat[0]*mat_Q).transpose();

        //reset
        weight = MatrixXf::Zero(1,k);
        mat_A_t_inv = (1.0/regularizer)*MatrixXf::Identity(k,k);//Init A

        //weight
        MatrixXf gradient = eta*y*(feature_mapping);//k*1
        // float fx = (weight*feature_mapping)(0,0);
        // MatrixXf gradient = 2*eta*y*(1-y*fx)*(feature_mapping); //squared hinge loss

        v_t = weight.transpose() + mat_A_t_inv*gradient;//k*1
        float coef = fun_h((feature_mapping.transpose()*v_t)(0,0))/((feature_mapping.transpose()*(mat_A_t_inv)*feature_mapping)(0,0)+0.0001);
        weight = (v_t-coef*(mat_A_t_inv)*feature_mapping).transpose();//1*k

        MatrixXf u = gradient/sigma_t;
        MatrixXf A_inv_dot_u = (mat_A_t_inv*u).transpose();//1*k
        mat_A_t_inv -= (mat_A_t_inv*u*u.transpose()*mat_A_t_inv)/(1+(A_inv_dot_u*u)(0,0));
    }
    else
    {
        MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma).transpose();
        
        MatrixXf feature_mapping = (mat_psi.transpose()*colsampleMat[0]*mat_Q).transpose();
        MatrixXf gradient = eta*y*(feature_mapping);//k*1
        // float fx = (weight*feature_mapping)(0,0);
        // MatrixXf gradient = 2*eta*y*(1-y*fx)*(feature_mapping); //squared hinge loss

        v_t = weight.transpose() + mat_A_t_inv*gradient;//k*1
        // cout<<"w:"<<weight<<endl;
        // cout<<"delta:"<<(mat_A_t.inverse()*gradient).transpose()<<endl;

        float coef = fun_h((feature_mapping.transpose()*v_t)(0,0))/((feature_mapping.transpose()*(mat_A_t_inv)*feature_mapping)(0,0)+0.0001);
        weight = (v_t-coef*(mat_A_t_inv)*feature_mapping).transpose();//1*k

        MatrixXf u = gradient/sigma_t;
        MatrixXf A_inv_dot_u = (mat_A_t_inv*u).transpose();//1*k
        mat_A_t_inv -= (mat_A_t_inv*u*u.transpose()*mat_A_t_inv)/(1+(A_inv_dot_u*u)(0,0));
    }
}

void SkeGD2nd::SkeGD_learning(MatrixXf dataX,MatrixXf dataY)
{
    int wheather_init = 0;
    int T0 = 0;
    for (int t = 0; t < dataX.rows(); t++)
    {
        MatrixXf xt = dataX.row(t);
        float yt = dataY(0,t);
        float f_t;
        
        // compute f_t(x_t)
        if(alpha.size()==0)
        {
            f_t = 0;
        }
        else
        {
            if (wheather_init == 0)
            {
                MatrixXf k_t = compute_K_SkeGD(SV_1,xt,sigma);
                MatrixXf mat_alpha = Map<MatrixXf, Eigen::Unaligned>(alpha.data(), 1,alpha.size());
                f_t = (mat_alpha*k_t)(0,0);
            }
            else
            {
                MatrixXf k_b = compute_K_SkeGD(xt,SV_1,sigma)*colsampleMat[0];//1*sm
                MatrixXf phi_t = (k_b*mat_Q).transpose();//k*1
                f_t = (weight*phi_t)(0,0);
            }
        }
        float hat_y_t = fun_sign(f_t);

        //count the number of errors
        if(abs(hat_y_t-yt)>0.0001 and wheather_init==1)
        {
            wrong++;
        }
        
        if(1-yt*f_t>0)
        {
            loss.push_back(yt*f_t);
        }
        else
        {
            loss.push_back(0);
        }


        if(yt*f_t<1)
        {
            T0++;
            if (SV_1.rows()<B)
            {
                alpha.push_back(eta*yt);
                if (SV_1.rows()==0)
                {
                    SV_1 = MatrixXf::Zero(1,xt.cols());
                    SV_1.row(0) = xt.row(0);
                }
                else
                {
                    SV_1.conservativeResize(SV_1.rows()+1, SV_1.cols());
                    SV_1.row(SV_1.rows()-1) = xt.row(0);
                }
            }
            else
            {
                if(SV_1.rows() == B and wheather_init == 0)
                {
                    Initialization_sketch(xt,yt);
                    wheather_init = 1;
                }
                else
                {
                    if (T0%rou == 1)//update
                    {
                        Update_sketch(xt,yt);                        
                    }
                    else
                    {
                        Update_sketch(xt,yt,0);
                    }
                }
            }
        }
    } 
    //cout<<"T0:"<<T0<<endl;
};

class NOGD
{
private:
    /* data */
public:
    NOGD(float eta,int B,int k,float sigma);
    ~NOGD();

    float eta;
    int B;
    int k;
    float sigma;

    MatrixXf SV;
    MatrixXf M;
    MatrixXf weight;
    MatrixXf alpha; 
    int wrong;
    std::vector<float> loss;

    void NOGD_learning(MatrixXf dataX,MatrixXf dataY);

    MatrixXf compute_K_SkeGD(MatrixXf X,MatrixXf Y,float sigma);
    int fun_sign(double x);
    Eigen::MatrixXf pinv(Eigen::MatrixXf  A);
};

NOGD::NOGD(float eta,int B,int k,float sigma)
{
    this->eta =eta;
    this->B = B;
    this->k = k;
    this->sigma = sigma;
    wrong = 0;
}

NOGD::~NOGD()
{
}

MatrixXf NOGD::compute_K_SkeGD(MatrixXf X,MatrixXf Y,float sigma)
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));
        }
    }
    return K;
}

int NOGD::fun_sign(double x)
{
    if(x>=0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

Eigen::MatrixXf NOGD::pinv(Eigen::MatrixXf  A)
{
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double  pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = min(row,col);
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(col,row);
    Eigen::MatrixXf singularValues_inv = svd.singularValues();
    Eigen::MatrixXf singularValues_inv_mat = Eigen::MatrixXf::Zero(col, row);
    for (long i = 0; i<k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i) 
    {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X=(svd.matrixV())*(singularValues_inv_mat)*(svd.matrixU().transpose());
 
    return X;
}

void NOGD::NOGD_learning(MatrixXf dataX,MatrixXf dataY)
{
    int wheather_init = 0;
    int T0 = 0;
    for (int t = 0; t < dataX.rows(); t++)
    {
        MatrixXf xt = dataX.row(t);
        float yt = dataY(0,t);
        float f_t;

        //compute f_t(x_t)
        if (alpha.cols()==0)
        {
            f_t = 0;
        }
        else
        {
            MatrixXf k_t = compute_K_SkeGD(SV,xt,sigma);//B*1
            if (wheather_init == 0)
            {
                f_t = (alpha*k_t)(0,0);// 1*B * B*1
            }
            else
            {
                MatrixXf nx_t = M*k_t;
                //cout<<"weight:"<<weight<<endl<<nx_t<<endl;
                f_t = (weight*nx_t)(0,0);
            }
        }
        //count the number of errors
        float hat_y_t = fun_sign(f_t);
        // cout<<f_t<<" "<<yt<<endl;
        if (abs(hat_y_t-yt)>0.0001 and wheather_init==1)
        {
            wrong++;
        }
        if(yt*f_t<1)
        {
            T0++;
            if (alpha.cols()<B)
            {
                if (alpha.cols()==0)
                {
                    alpha = MatrixXf::Zero(1,1);
                    alpha(0,0) = eta*yt;
                    SV = MatrixXf::Zero(1,xt.cols());
                    SV.row(0) = xt.row(0);
                }
                else
                {
                    alpha.conservativeResize(alpha.rows(), alpha.cols()+1);
                    alpha(0,alpha.cols()-1) =  eta*yt;
                    SV.conservativeResize(SV.rows()+1, SV.cols());
                    SV.row(SV.rows()-1) = xt.row(0);
                }
            }
            else
            {
                if(wheather_init == 0)
                {
                    MatrixXf k_hat = compute_K_SkeGD(SV,SV,sigma);

                    JacobiSVD<MatrixXf> svd(k_hat,ComputeThinU | ComputeThinV);
                    MatrixXf V = svd.matrixV(), U = svd.matrixU();
                    MatrixXf Vk = V.block(0,0,V.rows(),k);
                    MatrixXf Uk = U.block(0,0,U.rows(),k);
                    MatrixXf Dk = MatrixXf::Zero(k,k);
                    for (int j = 0; j < k; j++)
                    {
                        Dk(j,j) = pow(svd.singularValues()(j,0),-0.5);
                    }
                    M = Dk*(Vk.transpose());//k*k * k*B = k*B
                    wheather_init = 1;
                    weight = alpha*pinv(M);//1*B * B*k = 1*k
                    MatrixXf k_t = compute_K_SkeGD(SV,xt,sigma);//B*1
                    MatrixXf nx_t = M*k_t; //k*1

                    MatrixXf gradient = eta*yt*(nx_t.transpose());//hinge loss
                    // float fx = (weight*nx_t)(0,0);
                    // MatrixXf gradient = 2*eta*yt*(1-yt*fx)*(nx_t.transpose()); //squared hinge loss
                    weight = weight + gradient;
                }
                else
                {
                    MatrixXf k_t = compute_K_SkeGD(SV,xt,sigma);
                    MatrixXf nx_t = M*k_t; //k*1

                    MatrixXf gradient = eta*yt*(nx_t.transpose());//hinge loss
                    // float fx = (weight*nx_t)(0,0);
                    // MatrixXf gradient = 2*eta*yt*(1-yt*fx)*(nx_t.transpose()); //squared hinge loss
                    weight = weight + gradient;
                }
            }
        }
        
    }
};

MatrixXf compute_K(MatrixXf X,MatrixXf Y,float sigma)//计算核矩阵
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));
        }
    }
    return K;
}

class PROS_N_KONS
{
private:
    /* data */
public:
    PROS_N_KONS(float C,float sigma,float alpha,float beta);
    ~PROS_N_KONS();

    int wrong;
    float C;
    float sigma;
    float alpha;
    float beta;
    int j;
    MatrixXf weight;
    MatrixXf v;
    MatrixXf g;
    MatrixXf A_inv;
    MatrixXf S;
    MatrixXf SV;
    MatrixXf Sig_inv_j,U;//SVD
    MatrixXf phi_t;
    MatrixXf SKS_inv;

    float KORS(MatrixXf KMM,float lbd,float beta,float eps);
    MatrixXf update_inv(MatrixXf A_inv,MatrixXf x);
    void update(MatrixXf X,MatrixXf Y);
    float fun_sign(float x);
    float fun_pred(float x);
};

PROS_N_KONS::PROS_N_KONS(float C,float sigma,float alpha=1,float beta=1)
{
    this->C = C;
    this->sigma = sigma;
    this->alpha = alpha;
    this->beta = beta;
    SV.resize(0,0);
    j = 0;
    wrong = 0;
}

PROS_N_KONS::~PROS_N_KONS()
{
}

float PROS_N_KONS::fun_sign(float x)
{
    if(abs(x-0)<0.00001)
    {
        return 0;
    }
    else if(x>0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

float PROS_N_KONS::fun_pred(float x)
{
    if(x>0)
    {
        return 1;
    }
    else{
        return -1;
    }
}

MatrixXf PROS_N_KONS::update_inv(MatrixXf A_inv,MatrixXf x)//A_inv:(t-1)*(t-1) x:1*t
{
    if (A_inv.rows()==0)
    {
        MatrixXf D = x.col(x.cols()-1);
        D(0,0) = 1.0/D(0,0);
        return D;
    }
    else
    {
        MatrixXf B = x.block(0,0,1,x.cols()-1).transpose();//(t-1)*1
        MatrixXf C = B.transpose();//1*(t-1)
        MatrixXf D = x.col(x.cols()-1);
        MatrixXf mat_compl = D-C*A_inv*B;//1*1
        mat_compl(0,0) = 1.0/mat_compl(0,0);
        MatrixXf R0 = A_inv +A_inv*B*mat_compl*C*A_inv;
        MatrixXf R1 = -1*A_inv*B*mat_compl;
        MatrixXf R2 = -1*mat_compl*C*A_inv;
        MatrixXf R3 = mat_compl;
        MatrixXf R01(R0.rows(),R0.cols()+R1.cols());
        R01<<R0,R1;
        MatrixXf R23(R2.rows(),R2.cols()+R3.cols());
        R23<<R2,R3;
        MatrixXf R(R01.rows()+R23.rows(),R01.cols());
        R<<R01,
        R23;
        return R;
    }
}


float PROS_N_KONS::KORS(MatrixXf KMM,float lbd =1,float beta = 1e0,float eps = 0.5)
{
    if(S.rows()>0)
    {
        MatrixXf zero_S = MatrixXf::Zero(1,S.cols()+1);
        zero_S(0,S.cols()) = 1;
        S.conservativeResize(S.rows() + 1, S.cols()+1);
        S.row(S.rows()-1) = zero_S;
        S.col(S.cols()-1) = zero_S.transpose();
    }
    else
    {
        S.conservativeResize(S.rows() + 1, S.cols()+1);
        S(S.rows() - 1, S.cols()-1) = 1;
    }
    MatrixXf kS = (KMM.col(KMM.cols()-1)).transpose()*S;//1*t
    MatrixXf en = MatrixXf::Identity(kS.cols(),kS.cols()).row(kS.cols()-1); //1*t
    MatrixXf SKS_inv_temp = update_inv(SKS_inv,kS+lbd*en); //t*t
    float ktt = KMM(KMM.rows()-1,KMM.cols()-1);    
    float tau = ((1+eps)/lbd)*(ktt-(kS*SKS_inv_temp*kS.transpose())(0,0));
    float min_ele = min(beta*tau,float(1));
    float p = max(min_ele,float(0));
    std::random_device rd; 
    std::mt19937 gen(rd());
	std::bernoulli_distribution distribution(p);
    float z = distribution(gen);
    if (SV.rows()==0)
    {
        z = 1;
    }
    if(SV.rows()>100)
    {
        z = 0;
    }
    if(z)
    {
        S(S.rows()-1,S.cols()-1) = (1.0/(p));
        SKS_inv = update_inv(SKS_inv,(1.0/p)*(KMM.col(KMM.cols()-1)).transpose()*S+lbd*en);
    }
    else{
        // S = S.block(0,0,S.rows()-1,S.cols()-1);
        S.conservativeResize(S.rows()-1,S.cols()-1);
    }
    return z;
}

void PROS_N_KONS::update(MatrixXf X,MatrixXf Y)
{
    for (int t = 0; t < X.rows(); t++)
    {
        MatrixXf x = X.row(t);
        float y = Y(0,t);
        MatrixXf SV_temp;
        if(SV.rows()==0)
        {
            SV_temp.resize(1,x.cols());
            SV_temp.row(SV_temp.rows()-1) = x;//add x
        }
        else
        {
            SV_temp = SV;
            SV_temp.conservativeResize(SV_temp.rows() + 1, SV_temp.cols());
            SV_temp.row(SV_temp.rows()-1) = x;//add x
        }
        MatrixXf KMM = compute_K(SV_temp,SV_temp,sigma);
        float z = KORS(KMM,alpha,beta);
        if(z)
        {
            j++;
            SV = SV_temp;//add x
            JacobiSVD<MatrixXf> svd(KMM,ComputeThinU | ComputeThinV);
            U = svd.matrixU();
            Sig_inv_j = MatrixXf::Zero(KMM.rows(),KMM.rows());
            for (int i = 0; i < KMM.rows(); i++)
            {
                Sig_inv_j(i,i) = 1.0/(sqrt(svd.singularValues()(i,0))+0.00001);
            }
            //cout<<"U"<<endl<<U<<endl<<"siginv"<<endl<<Sig_inv_j<<endl;
            A_inv = (1.0/alpha)*MatrixXf::Identity(j,j);
            weight = MatrixXf::Zero(j,1);//j*1
            phi_t = MatrixXf::Zero(j,1);//j*1
        }
        else
        {
            MatrixXf Kt = compute_K(SV,x,sigma); //j*1
            phi_t = Sig_inv_j*(U.transpose())*Kt; //j*1
            // cout<<"weight:"<<weight.transpose()<<endl;
            // cout<<"delta:"<<(A_inv*g).transpose()<<endl;
            v = weight-(A_inv*g); //2nd j*1
            float h = fun_sign((phi_t.transpose()*v)(0,0))*max(abs((phi_t.transpose()*v)(0,0))-C,float(0));
            float coef = float(h)/((phi_t.transpose()*A_inv*phi_t)(0,0)+0.0001);
            weight = v - coef*A_inv*phi_t; //j*1
        }
        //cout<<"phi:"<<phi_t.transpose()<<endl;
        float yt_hat = (weight.transpose()*phi_t)(0,0);
        float yt_hat_lable = fun_pred(yt_hat);//
        //cout<<"yt_hat:"<<yt_hat<<" y:"<<y<<endl;
        if (abs(yt_hat_lable-y)>0.0001)
        {
            wrong++;
        }
        
        if(y*yt_hat<1)
        {
            g = -y*phi_t;//j*1
        }
        else
        {
            g = MatrixXf::Zero(j,1);
        }
        MatrixXf u = g/4;
        MatrixXf A_inv_dot_u = (A_inv*u).transpose(); //1*j
        A_inv -= (A_inv_dot_u.transpose()*A_inv_dot_u)/(1+(A_inv_dot_u*u)(0,0));//update
    } 
};

void removeRow(Eigen::MatrixXf& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = int(matrix.rows()) - 1;
    unsigned int numCols = int(matrix.cols());

    if (rowToRemove < numRows)
        matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);
    matrix.conservativeResize(numRows, numCols);
}
void removeColumn(Eigen::MatrixXf& matrix, unsigned int colToRemove)
{
    unsigned int numRows = int(matrix.rows());
    unsigned int numCols = int(matrix.cols()) - 1;
    if (colToRemove < numCols)
        matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);
    matrix.conservativeResize(numRows, numCols);
}

class RBP
{
private:
    /* data */
public:
    RBP(int B,float sigma);
    ~RBP();

    int B;
    float sigma;
    MatrixXf alpha;
    MatrixXf SV;
    int wrong;

    void RBP_learning(MatrixXf dataX,MatrixXf dataY);
    MatrixXf compute_K(MatrixXf X,MatrixXf Y,float sigma);
    int fun_sign(double x);
};

RBP::RBP(int B,float sigma)
{
    this->B = B;
    this->sigma = sigma;
    wrong = 0;
}

RBP::~RBP()
{
}

MatrixXf RBP::compute_K(MatrixXf X,MatrixXf Y,float sigma)
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));
        }
    }
    return K;
}

int RBP::fun_sign(double x)
{
    if(x>=0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

void RBP::RBP_learning(MatrixXf dataX,MatrixXf dataY)
{
    for (int t = 0; t < dataX.rows(); t++)
    {
        MatrixXf xt = dataX.row(t);
        float yt = dataY(0,t);
        float f_t;

        //compute f_t(x_t)
        if (alpha.cols()==0)
        {
            f_t = 0;
        }
        else
        {
            MatrixXf k_t = compute_K(SV,xt,sigma);//B*1
            f_t = (alpha*k_t)(0,0);
        }
        //count the number of errors
        float hat_y_t = fun_sign(f_t);
        //cout<<f_t<<" "<<yt<<endl;
        if (abs(hat_y_t-yt)>0.0001)
        {
            wrong++;
            if(alpha.cols()<B)
            {
                if (alpha.cols()==0)
                {
                    alpha = MatrixXf::Zero(1,1);
                    alpha(0,0) = yt;
                    SV = MatrixXf::Zero(1,xt.cols());
                    SV.row(0) = xt.row(0);
                }
                else
                {
                    alpha.conservativeResize(alpha.rows(), alpha.cols()+1);
                    alpha(0,alpha.cols()-1) =  yt;
                    SV.conservativeResize(SV.rows()+1, SV.cols());
                    SV.row(SV.rows()-1) = xt.row(0);
                }
            }
            else
            {
                std::vector<int> perm_t;
                for (int i = 0; i < B; i++)
                {
                    perm_t.push_back(i);
                }
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(perm_t.begin(), perm_t.end(), g);
                int idx = perm_t[0];
                std::vector<int> subset;
                for (int i = 0; i < B-1; i++)
                {
                    if(i!=idx)
                    {
                        subset.push_back(i);
                    }
                }
                removeColumn(alpha,idx);
                removeRow(SV,idx);
            }
        }

    }
};

class BPAS
{
private:
    /* data */
public:
    BPAS(int B,float C,float sigma);
    ~BPAS();

    float C;
    int B;
    int wrong;
    float sigma;
    MatrixXf alpha;
    MatrixXf SV;

    void BPAS_learning(MatrixXf dataX,MatrixXf dataY);
    int fun_sign(double x);
    MatrixXf compute_K(MatrixXf X,MatrixXf Y,float sigma);
};

BPAS::BPAS(int B,float C,float sigma)
{
    this->B = B;
    this->C = C;
    this->sigma = sigma;
    wrong = 0;
}

BPAS::~BPAS()
{
}

int BPAS::fun_sign(double x)
{
    if(x>=0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

MatrixXf BPAS::compute_K(MatrixXf X,MatrixXf Y,float sigma)
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));
        }
    }
    return K;
}

void BPAS::BPAS_learning(MatrixXf dataX,MatrixXf dataY)
{
    int star = 0;
    float star_alpha;
    for (int t = 0; t < dataX.rows(); t++)
    {
        MatrixXf xt = dataX.row(t);//1*t
        float yt = dataY(0,t);
        float f_t;

        //compute f_t(x_t)
        if (alpha.cols()==0)
        {
            f_t = 0;
        }
        else
        {
            MatrixXf k_t = compute_K(SV,xt,sigma);//B*1
            f_t = (alpha*k_t)(0,0);
        }
        //count the number of errors
        float hat_y_t = fun_sign(f_t);
        if (abs(hat_y_t-yt)>0.0001)
        {
            wrong++;
        }
        //cout<<f_t<<" "<<yt<<endl;


        if(yt*f_t<1)
        {
            float l_t = 1-yt*f_t;
            if(alpha.cols()<B)
            {
                float s_t = 1;
                float tau_t = min(C,l_t/s_t);
                if (alpha.cols()==0)
                {
                    alpha = MatrixXf::Zero(1,1);
                    alpha(0,0) = yt*tau_t;
                    SV = MatrixXf::Zero(1,xt.cols());
                    SV.row(0) = xt.row(0);
                }
                else
                {
                    alpha.conservativeResize(alpha.rows(), alpha.cols()+1);
                    alpha(0,alpha.cols()-1) =  yt*tau_t;
                    SV.conservativeResize(SV.rows()+1, SV.cols());
                    SV.row(SV.rows()-1) = xt.row(0);
                }
            }
            else
            {
                float Q_star = 10000000;
                MatrixXf alpha_star = alpha;
                MatrixXf SV_star = SV;
                float k_tt = 1;
                float tau_t = min(C,l_t/k_tt);
                MatrixXf k_t = compute_K(SV,xt,sigma);//B*1
                for (int r = 0; r < B; r++)
                {
                    float k_rt = k_t(r,0);
                    float alpha_r = alpha(0,r);
                    float beta_t = alpha_r*k_rt+tau_t*yt;
                    float distance_f_rt = pow(alpha_r,2)+pow(beta_t,2)-2*alpha_r*beta_t*k_rt;
                    float f_rt = f_t-alpha_r*k_rt+beta_t;
                    float l_rt = max(float(0),1-yt*f_rt);
                    float Q_r = 0.5*distance_f_rt+C*l_rt;
                    if (Q_r<Q_star)
                    {
                        Q_star = Q_r;
                        star = r;
                        star_alpha = beta_t;
                    } 
                }
                alpha_star = alpha;
                alpha_star(0,star) = star_alpha;
                SV_star = SV;
                SV_star.row(star) = xt;
                alpha = alpha_star;
                SV = SV_star;
            }
        }
    }
};

class Projectron
{
private:
    /* data */
public:
    Projectron(int B,float sigma);
    ~Projectron();

    int B;
    float sigma;
    MatrixXf alpha;
    MatrixXf SV;
    int wrong;

    MatrixXf compute_K(MatrixXf X,MatrixXf Y,float sigma);
    int fun_sign(double x);

    void Projectron_learning(MatrixXf dataX,MatrixXf dataY);
};

Projectron::Projectron(int B,float sigma)
{
    this->B = B;
    this->sigma = sigma;
    wrong = 0;
}

Projectron::~Projectron()
{
}

MatrixXf Projectron::compute_K(MatrixXf X,MatrixXf Y,float sigma)
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));
        }
    }
    return K;
}

int Projectron::fun_sign(double x)
{
    if(x>=0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

void Projectron::Projectron_learning(MatrixXf dataX,MatrixXf dataY)
{
    MatrixXf Kid;
    MatrixXf K_t_inver;
    float kxx;
    for (int t = 0; t < dataX.rows(); t++)
    {
        MatrixXf xt = dataX.row(t);
        float yt = dataY(0,t);
        float f_t;
        //compute f_t(x_t)
        if (alpha.cols()==0)
        {
            f_t = 0;
        }
        else
        {
            MatrixXf k_t = compute_K(SV,xt,sigma);//B*1
            f_t = (alpha*k_t)(0,0);
        }

        //count the number of errors
        float hat_y_t = fun_sign(f_t);
        //cout<<f_t<<" "<<yt<<endl;
        if (abs(hat_y_t-yt)>0.0001)
        {
            wrong++;
        }

        if(alpha.cols()==0)
        {
            if(abs(hat_y_t-yt)>0.0001)
            {
                alpha = MatrixXf::Zero(1,1);
                alpha(0,0) = yt;
                SV = MatrixXf::Zero(1,xt.cols());
                SV.row(0) = xt.row(0);
                Kid = compute_K(xt,xt,sigma);
                kxx = Kid(0,0);
                K_t_inver = Kid.inverse();
            }
        }
        else
        {
            if(abs(hat_y_t-yt)>0.0001)
            {
                MatrixXf k_t = compute_K(SV,xt,sigma);//B*1
                MatrixXf d_star = K_t_inver*k_t;//B*1
                float norm_delta_t = sqrt(kxx-(k_t.transpose()*d_star)(0,0));
                if(SV.rows()==B)
                {
                    alpha += yt*d_star.transpose();
                }
                else
                {
                    alpha.conservativeResize(alpha.rows(), alpha.cols()+1);
                    alpha(0,alpha.cols()-1) =  yt;
                    SV.conservativeResize(SV.rows()+1, SV.cols());
                    SV.row(SV.rows()-1) = xt.row(0);
                    int size_SV = SV.rows();
                    MatrixXf temp = K_t_inver;//B*B
                    temp.conservativeResize(size_SV,size_SV);
                    MatrixXf zero_left = MatrixXf::Zero(temp.rows(),1);
                    temp.col(temp.cols()-1) = zero_left;
                    temp.row(temp.rows()-1) = zero_left.transpose();
                    MatrixXf d_til = d_star;
                    d_til.conservativeResize(d_til.rows()+1,d_til.cols());
                    d_til(d_til.rows()-1,0) = -1;//(B+1)*1
                    K_t_inver = temp+d_til*d_til.transpose()/pow(norm_delta_t,2);
                }
            }
        }
        
    }
};

float max_abs_matrix(MatrixXf A)//1*t
{
    float max = -100000000;
    for (int i = 0; i < A.cols(); i++)
    {
        if(abs(A(0,i))>max)
        {
            max = abs(A(0,i));
        }
    }
    return max;   
}

class BOGD
{
private:
    /* data */
public:
    BOGD(float eta,int B,float lambda,float gama,float sigma);
    ~BOGD();

    float eta;
    float lambda;
    float gama;
    float sigma;
    int B;
    MatrixXf SV;
    MatrixXf weight;
    MatrixXf alpha; //1*B
    int wrong;

    void BOGD_learning(MatrixXf dataX,MatrixXf dataY);
    MatrixXf compute_K(MatrixXf X,MatrixXf Y,float sigma);
    int fun_sign(double x);
    
};

BOGD::BOGD(float eta,int B,float lambda,float gama,float sigma)
{
    this->B = B;
    this->eta = eta;
    this->lambda = lambda;
    this->gama = gama;
    this->sigma = sigma;
    wrong = 0;
}

BOGD::~BOGD()
{
}

MatrixXf BOGD::compute_K(MatrixXf X,MatrixXf Y,float sigma)
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));
        }
    }
    return K;
}

int BOGD::fun_sign(double x)
{
    if(x>=0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

void BOGD::BOGD_learning(MatrixXf dataX,MatrixXf dataY)
{
    int T0 = 0;
    for (int t = 0; t < dataX.rows(); t++)
    {
        MatrixXf xt = dataX.row(t);
        float yt = dataY(0,t);
        float f_t;

        //compute f_t(x_t)
        if (alpha.cols()==0)
        {
            f_t = 0;
        }
        else
        {
            MatrixXf k_t = compute_K(SV,xt,sigma);//B*1
            f_t = (alpha*k_t)(0,0);
        }
        //count the number of errors
        float hat_y_t = fun_sign(f_t);
        // cout<<f_t<<" "<<yt<<endl;
        if (abs(hat_y_t-yt)>0.0001)
        {
            wrong++;
        }
        if(yt*f_t<1)
        {
            if(alpha.cols()<B)
            {
                float ell_gradient = -1;
                if (alpha.cols()==0)
                {
                    alpha = MatrixXf::Zero(1,1);
                    alpha(0,0) = yt*(-eta*ell_gradient);
                    SV = MatrixXf::Zero(1,xt.cols());
                    SV.row(0) = xt.row(0);
                }
                else
                {
                    alpha.conservativeResize(alpha.rows(), alpha.cols()+1);
                    alpha(0,alpha.cols()-1) =  yt*(-eta*ell_gradient);
                    SV.conservativeResize(SV.rows()+1, SV.cols());
                    SV.row(SV.rows()-1) = xt.row(0);
                }
            }
            else
            {
                std::vector<int> perm_t;
                for (int i = 0; i < B; i++)
                {
                    perm_t.push_back(i);
                }
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(perm_t.begin(), perm_t.end(), g);
                int idx = perm_t[0];
                std::vector<int> subset;
                for (int i = 0; i < B-1; i++)
                {
                    if(i!=idx)
                    {
                        subset.push_back(i);
                    }
                }

                alpha = (B*(1-lambda*eta)/(B-1))*alpha;
                removeColumn(alpha,idx);
                removeRow(SV,idx);

                float ell_gradient=-1;
                alpha.conservativeResize(alpha.rows(), alpha.cols()+1);
                alpha(0,alpha.cols()-1) =  yt*(-eta*ell_gradient);
                alpha = alpha*min((eta*gama/(max_abs_matrix(alpha))),float(1));
                SV.conservativeResize(SV.rows()+1, SV.cols());
                SV.row(SV.rows()-1) = xt.row(0);
            }
        }
        else
        {
            alpha = alpha*(1-eta*lambda);
        }
    }
};

class SkeGD2nd_origin
{
private:
    /* data */
public:
    SkeGD2nd_origin(int sp,int sm,float eta,int k,int B,int rou,float sigma,int d,float regularizer):sketch(B,sp,d)
    {
        this->sp = sp;
        this->sm = sm;
        this->eta=eta;
        this-> k = k;
        this->B = B;
        this->rou = rou;
        this->sigma = sigma;
        this->d = d;
        this->regularizer = regularizer;
        sigma_t = 1;
        wrong = 0;
        batch_wrong = 0;
    };
    ~SkeGD2nd_origin();

    int sp;//sketch size
    int sm;//sample size
    float eta;//stepsize
    int k;//SVD rank
    int B;//budget
    int rou;//updae cycle
    int d;
    float sigma;
    float sigma_t;
    float regularizer;

    MatrixXf SV_1;
    MatrixXf mat_phi_pm;
    MatrixXf mat_phi_pp;
    MatrixXf mat_Q;
    MatrixXf weight;
    MatrixXf v_t;
    MatrixXf mat_A_t_inv;
    SJLT sketch;
    std::vector<MatrixXf> colsampleMat;
    std::vector<float> alpha; 
    int wrong;
    int batch_wrong;

    void Initialization_sketch(MatrixXf x,float y);
    void Update_sketch(MatrixXf x,float y,int flag=1);
    void SkeGD_learning(MatrixXf dataX,MatrixXf dataY);
    void Batch_learning(MatrixXf dataX,MatrixXf dataY);



    int fun_sign(double x); 
    int fun_sign2(float x);
    MatrixXf compute_K_SkeGD(MatrixXf X,MatrixXf Y,float sigma);
    std::vector<MatrixXf> colsampling(MatrixXf mat,int col);
    Eigen::MatrixXf pinv(Eigen::MatrixXf  A);
    float fun_h(float x,float C=1);

    std::vector<float> loss;

};

SkeGD2nd_origin::~SkeGD2nd_origin()
{
}

MatrixXf SkeGD2nd_origin::compute_K_SkeGD(MatrixXf X,MatrixXf Y,float sigma)
{
    MatrixXf mat_id=X;
    MatrixXf mat_SV=Y;
    
    MatrixXf K = MatrixXf::Zero(X.rows(),Y.rows());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < Y.rows(); j++)
        {
            K(i,j) = exp(-((mat_id.row(i)-mat_SV.row(j)).squaredNorm()/(2*sigma*sigma)));
            //K(i,j) = (mat_id.row(i)*mat_SV.row(j).transpose())(0,0);
        }
    }
    return K;
}

int SkeGD2nd_origin::fun_sign(double x)
{
    if(x>=0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int SkeGD2nd_origin::fun_sign2(float x)
{
    if(abs(x-0)<0.0001)
    {
        return 0;
    }
    else if (x>0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

float SkeGD2nd_origin::fun_h(float x,float C)
{
    float anw = 0;
    if((abs(x)-C)>anw)
    {
        anw = abs(x)-C;
    }
    return fun_sign2(x)*anw;
}

std::vector<MatrixXf> SkeGD2nd_origin::colsampling(MatrixXf mat,int col)
{
    std::vector<MatrixXf> output;
    MatrixXf mat_I = MatrixXf::Identity(mat.cols(),mat.cols());
    MatrixXf mat_Sm = MatrixXf::Zero(mat.cols(),col);
    
    std::vector<int> index;
    for (int i = 0; i < mat.cols(); i++)
    {
        index.push_back(i);
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(index.begin(), index.end(), g);

    for (int i = 0; i < col; i++)
    {
        mat_Sm.col(i) = mat_I.col(index[i]);
    }

    output.push_back(mat_Sm);
    output.push_back(mat*mat_Sm);
    return output;
}

Eigen::MatrixXf SkeGD2nd_origin::pinv(Eigen::MatrixXf  A)
{
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double  pinvtoler = 1.e-8; //tolerance
    int row = A.rows();
    int col = A.cols();
    int k = min(row,col);
    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(col,row);
    Eigen::MatrixXf singularValues_inv = svd.singularValues();
    Eigen::MatrixXf singularValues_inv_mat = Eigen::MatrixXf::Zero(col, row);
    for (long i = 0; i<k; ++i) {
        if (singularValues_inv(i) > pinvtoler)
            singularValues_inv(i) = 1.0 / singularValues_inv(i);
        else singularValues_inv(i) = 0;
    }
    for (long i = 0; i < k; ++i) 
    {
        singularValues_inv_mat(i, i) = singularValues_inv(i);
    }
    X=(svd.matrixV())*(singularValues_inv_mat)*(svd.matrixU().transpose());
 
    return X;
}

void SkeGD2nd_origin::Initialization_sketch(MatrixXf x,float y)
{
    MatrixXf K_B = compute_K_SkeGD(SV_1,SV_1,sigma); 
    sketch.generation();
    //cout<<"K_B:"<<endl<<K_B<<endl;
    MatrixXf mat_C_p = K_B*sketch.sketch_matrix;

    colsampleMat = colsampling(K_B,sm);
    MatrixXf mat_C_m = colsampleMat[1];
    mat_phi_pm = sketch.sketch_matrix.transpose()*mat_C_m;
    mat_phi_pp = sketch.sketch_matrix.transpose()*mat_C_p;

    
    JacobiSVD<MatrixXf> svd(mat_phi_pp,ComputeThinU | ComputeThinV);
    MatrixXf V = svd.matrixV(), U = svd.matrixU();
    MatrixXf Vk = V.block(0,0,V.rows(),k);
    MatrixXf Uk = U.block(0,0,U.rows(),k);
    MatrixXf S_sqrt = MatrixXf::Zero(k,k);
    for (int j = 0; j < k; j++)
    {
        S_sqrt(j,j) = sqrt(svd.singularValues()(j,0));
    }
    mat_Q = pinv(mat_phi_pm)*Vk*S_sqrt;

    MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma);
    MatrixXf feature_mapping = (mat_psi*colsampleMat[0]*mat_Q).transpose();//k*1

    MatrixXf mat_alpha = Map<MatrixXf, Eigen::Unaligned>(alpha.data(), 1,alpha.size());
    weight = mat_alpha*(mat_psi.transpose())*pinv(feature_mapping);
    mat_A_t_inv = (1.0/regularizer)*MatrixXf::Identity(k,k);
    MatrixXf gradient = eta*y*(feature_mapping);//k*1
    v_t = weight.transpose() + mat_A_t_inv*gradient;//k*1
    float coef = fun_h((feature_mapping.transpose()*v_t)(0,0))/(feature_mapping.transpose()*(mat_A_t_inv)*feature_mapping)(0,0);
    weight = (v_t-coef*(mat_A_t_inv)*feature_mapping).transpose();

    MatrixXf u = gradient/(sigma_t);
    MatrixXf A_inv_dot_u = (mat_A_t_inv*u).transpose();//1*k
    mat_A_t_inv -= (mat_A_t_inv*u*u.transpose()*mat_A_t_inv)/(1+(A_inv_dot_u*u)(0,0));
}

void SkeGD2nd_origin::Update_sketch(MatrixXf x,float y,int flag)
{
    if(flag==1)
    {
        MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma).transpose();//B*1
        SJLT ske_sub(1,sp,d);
        ske_sub.generation();
        MatrixXf mat_R_pm = (ske_sub.sketch_matrix.transpose())*(mat_psi.transpose()*colsampleMat[0]);
        mat_phi_pm = mat_phi_pm+mat_R_pm;
        MatrixXf mat_R_pp = ske_sub.sketch_matrix.transpose()*(mat_psi.transpose()*sketch.sketch_matrix);
        MatrixXf xi = compute_K_SkeGD(x,x,sigma);
        MatrixXf mat_T_pp = xi(0,0)*ske_sub.sketch_matrix.transpose()*ske_sub.sketch_matrix;
        mat_phi_pp = mat_phi_pp+mat_R_pp+mat_R_pp.transpose()+mat_T_pp;


        JacobiSVD<MatrixXf> svd(mat_phi_pp,ComputeThinU | ComputeThinV);
        MatrixXf V = svd.matrixV(), U = svd.matrixU();
        MatrixXf Vk = V.block(0,0,V.rows(),k);
        MatrixXf Uk = U.block(0,0,U.rows(),k);
        MatrixXf S_sqrt = MatrixXf::Zero(k,k);
        for (int j = 0; j < k; j++)
        {
            S_sqrt(j,j) = sqrt(svd.singularValues()(j,0));
        }

        mat_Q = pinv(mat_phi_pm)*Vk*S_sqrt;

        MatrixXf feature_mapping = (mat_psi.transpose()*colsampleMat[0]*mat_Q).transpose();

        weight = MatrixXf::Zero(1,k);
        mat_A_t_inv = (1.0/regularizer)*MatrixXf::Identity(k,k);

        MatrixXf gradient = eta*y*(feature_mapping);//k*1
        // float fx = (weight*feature_mapping)(0,0);
        // MatrixXf gradient = 2*eta*y*(1-y*fx)*(feature_mapping); //squared hinge loss
        
        v_t = weight.transpose() + mat_A_t_inv*gradient;//k*1
        float coef = fun_h((feature_mapping.transpose()*v_t)(0,0))/((feature_mapping.transpose()*(mat_A_t_inv)*feature_mapping)(0,0)+0.0001);
        weight = (v_t-coef*(mat_A_t_inv)*feature_mapping).transpose();

        MatrixXf u = gradient/sigma_t;
        MatrixXf A_inv_dot_u = (mat_A_t_inv*u).transpose();//1*k
        mat_A_t_inv -= (mat_A_t_inv*u*u.transpose()*mat_A_t_inv)/(1+(A_inv_dot_u*u)(0,0));
    }
    else
    {
        MatrixXf mat_psi = compute_K_SkeGD(x,SV_1,sigma).transpose();

        MatrixXf feature_mapping = (mat_psi.transpose()*colsampleMat[0]*mat_Q).transpose();
        MatrixXf gradient = eta*y*(feature_mapping);//k*1
        // float fx = (weight*feature_mapping)(0,0);
        // MatrixXf gradient = 2*eta*y*(1-y*fx)*(feature_mapping); //squared hinge loss

        v_t = weight.transpose() + mat_A_t_inv*gradient;//k*1
        // cout<<"w:"<<weight<<endl;
        // cout<<"delta:"<<(mat_A_t.inverse()*gradient).transpose()<<endl;

        float coef = fun_h((feature_mapping.transpose()*v_t)(0,0))/((feature_mapping.transpose()*(mat_A_t_inv)*feature_mapping)(0,0)+0.0001);
        weight = (v_t-coef*(mat_A_t_inv)*feature_mapping).transpose();//1*k

        MatrixXf u = gradient/sigma_t;
        MatrixXf A_inv_dot_u = (mat_A_t_inv*u).transpose();//1*k
        mat_A_t_inv -= (mat_A_t_inv*u*u.transpose()*mat_A_t_inv)/(1+(A_inv_dot_u*u)(0,0));
    }
}

void SkeGD2nd_origin::SkeGD_learning(MatrixXf dataX,MatrixXf dataY)
{
    int wheather_init = 0;
    int T0 = 0;
    for (int t = 0; t < dataX.rows(); t++)
    {
        MatrixXf xt = dataX.row(t);
        float yt = dataY(0,t);
        float f_t;
        
        // compute f_t(x_t)
        if(alpha.size()==0)
        {
            f_t = 0;
        }
        else
        {
            if (wheather_init == 0)
            {
                MatrixXf k_t = compute_K_SkeGD(SV_1,xt,sigma);
                MatrixXf mat_alpha = Map<MatrixXf, Eigen::Unaligned>(alpha.data(), 1,alpha.size());
                f_t = (mat_alpha*k_t)(0,0);
            }
            else
            {
                MatrixXf k_b = compute_K_SkeGD(xt,SV_1,sigma)*colsampleMat[0];//1*sm
                MatrixXf phi_t = (k_b*mat_Q).transpose();//k*1
                f_t = (weight*phi_t)(0,0);
                // cout<<"weight:"<<weight<<endl;
                // cout<<"phi:"<<phi_t.transpose()<<endl;
            }
        }
        float hat_y_t = fun_sign(f_t);
        //cout<<"f_t:"<<f_t<<" y:"<<yt<<endl;

        //count the number of errors
        if(abs(hat_y_t-yt)>0.0001 and wheather_init==1)
        {
            wrong++;
        }
        
        if(1-yt*f_t>0)
        {
            loss.push_back(yt*f_t);
        }
        else
        {
            loss.push_back(0);
        }


        if(yt*f_t<1)
        {
            T0++;
            if (SV_1.rows()<B)
            {
                alpha.push_back(eta*yt);
                if (SV_1.rows()==0)
                {
                    SV_1 = MatrixXf::Zero(1,xt.cols());
                    SV_1.row(0) = xt.row(0);
                }
                else
                {
                    SV_1.conservativeResize(SV_1.rows()+1, SV_1.cols());
                    SV_1.row(SV_1.rows()-1) = xt.row(0);
                }
            }
            else
            {
                if(SV_1.rows() == B and wheather_init == 0)
                {
                    Initialization_sketch(xt,yt);
                    wheather_init = 1;
                }
                else
                {
                    if (T0%rou == 1)
                    {
                        Update_sketch(xt,yt);
                    }
                    else
                    {
                        Update_sketch(xt,yt,0);
                    }
                }
            }
        }
    } 
    //cout<<"T0:"<<T0<<endl;
};

}

#endif



