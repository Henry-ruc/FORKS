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
#include "OKL_fun.hpp"
#define PI M_PI
using namespace std;
using namespace chrono;  
using namespace Eigen;
using namespace Eigen::internal;  
using namespace Eigen::Architecture;

std::vector<float> generat_para(float begin,float end,int n)
{
    float d = (end-begin)/float(n-1);
    std::vector<float> anw;
    float delta  = begin;
    for (int i = 0; i < n; i++)
    {
        anw.push_back(delta);
        delta+=d;
    }
    return anw;
}

int main(int argc, char const *argv[])
{
    srand((unsigned)time(NULL));


    ifstream inFile("/Kuairec/KuaiRuc_version_2.csv");
    string lineStr;
    vector<vector<float> > strArray;
    while (getline(inFile, lineStr)) {
        //cout << lineStr << endl;
        stringstream ss(lineStr);
        string str;
        vector<float> lineArray;
        while (getline(ss, str, ',')){

        float str_float;
        istringstream istr(str);
        istr >> str_float;
        lineArray.push_back(str_float);
        }
        strArray.push_back(lineArray);
    }

    int N = 4494578;
    int dim_x = 213;
    MatrixXf dataX = MatrixXf::Zero(N,dim_x-1);
    std::vector<float> dataY;
    for (int i = 0; i < N; i++)
    {
        for (int j = 1; j < dim_x; j++)
        {
            dataX(i,j-1) = strArray[i][j];
        }
        dataY.push_back(strArray[i][0]);
    }
    MatrixXf Y = Map<MatrixXf, Eigen::Unaligned>(dataY.data(), 1,dataY.size());
    


    float begin_B = 100;
    float end_B = 500;
    int n_B = 9;
    std::vector<float> B_set = generat_para(begin_B,end_B,n_B);
    ofstream outFile;
    outFile.open("/kuai_B.csv", ios::out); 

    for (int B_loop = 0; B_loop < n_B; B_loop++)
    {
        std::vector<float> time_1nd;
        std::vector<float> wrong_rate_1nd;
        std::vector<float> time_2nd;
        std::vector<float> wrong_rate_2nd;
        std::vector<float> time_NOGD;
        std::vector<float> wrong_rate_NOGD;
        std::vector<float> time_PROS;
        std::vector<float> wrong_rate_PROS;
        std::vector<float> time_BOGD;
        std::vector<float> wrong_rate_BOGD;
        for (int t = 0; t < 1; t++)
        {
            //SkeGD setting
            int B = B_set[B_loop];
            int sp = 3*B/4;
            int sm = 0.2*sp;
            float eta = 0.2;
            int k = 0.1*B;
            int d = 1;
            float regularizer = 0.01;
            int rou = floor(0.005*(N-B));
            //else
            int Budget = B_set[B_loop]; 
            float C_BPAS = 1;
            float lambda = 1.0/(N*N);
            float gamma = 10;
            float sigma = 25;


            SkeGD test_1nd(sp,sm,eta,k,B,rou,sigma,d);
            SkeGD2nd test_2nd(sp,sm,eta,k,B,rou,sigma,d,regularizer);
            NOGD test_NOGD(eta,B,k,sigma);
            PROS_N_KONS test_PROS(C_BPAS,sigma);
            RBP test_RBP(Budget,sigma);
            BPAS test_BPAS(Budget,C_BPAS,sigma);
            Projectron test_proj(Budget,sigma);
            BOGD test_BOGD(eta,Budget,lambda,gamma,sigma);

            system_clock::time_point beg_t_1nd = system_clock::now();
            test_1nd.SkeGD_learning(dataX,Y);
            system_clock::time_point end_t_1nd = system_clock::now();
            duration<double> diff_1nd = end_t_1nd - beg_t_1nd;
            time_1nd.push_back(diff_1nd.count());
            wrong_rate_1nd.push_back(float(test_1nd.wrong)/(N));

            system_clock::time_point beg_t_2nd = system_clock::now();
            test_2nd.SkeGD_learning(dataX,Y);
            system_clock::time_point end_t_2nd = system_clock::now();
            duration<double> diff_2nd = end_t_2nd - beg_t_2nd;
            time_2nd.push_back(diff_2nd.count());
            wrong_rate_2nd.push_back(float(test_2nd.wrong)/(N));

            system_clock::time_point beg_t_NOGD = system_clock::now();
            test_NOGD.NOGD_learning(dataX,Y);
            system_clock::time_point end_t_NOGD = system_clock::now();
            duration<double> diff_NOGD = end_t_NOGD - beg_t_NOGD;
            time_NOGD.push_back(diff_NOGD.count());
            wrong_rate_NOGD.push_back(float(test_NOGD.wrong)/(N));

            system_clock::time_point beg_t_PROS = system_clock::now();
            test_PROS.update(dataX,Y);
            system_clock::time_point end_t_PROS = system_clock::now();
            duration<double> diff_PROS = end_t_PROS - beg_t_PROS;
            time_PROS.push_back(diff_PROS.count());
            wrong_rate_PROS.push_back(float(test_PROS.wrong)/(N));

            system_clock::time_point beg_t_BOGD = system_clock::now();
            test_BOGD.BOGD_learning(dataX,Y);
            system_clock::time_point end_t_BOGD = system_clock::now();
            duration<double> diff_BOGD = end_t_BOGD - beg_t_BOGD;
            time_BOGD.push_back(diff_BOGD.count());
            wrong_rate_BOGD.push_back(float(test_BOGD.wrong)/(N));
        }
        outFile << B_set[B_loop] << endl;

        //SkeGD
        double sum = std::accumulate(std::begin(time_1nd), std::end(time_1nd), 0.0);  
        double mean_time_1nd =  sum / time_1nd.size(); 
        sum = std::accumulate(std::begin(wrong_rate_1nd), std::end(wrong_rate_1nd), 0.0);  
        double mean_1nd =  sum / wrong_rate_1nd.size(); 
        outFile << mean_time_1nd << ',' << mean_1nd << endl;
        //SkeGD2nd
        sum = std::accumulate(std::begin(time_2nd), std::end(time_2nd), 0.0);  
        double mean_time_2nd =  sum / time_2nd.size();
        sum = std::accumulate(std::begin(wrong_rate_2nd), std::end(wrong_rate_2nd), 0.0);  
        double mean_2nd =  sum / wrong_rate_2nd.size();
        outFile << mean_time_2nd << ',' << mean_2nd << endl;
        //NOGD
        sum = std::accumulate(std::begin(time_NOGD), std::end(time_NOGD), 0.0);  
        double mean_time_NOGD =  sum / time_NOGD.size(); 
        sum = std::accumulate(std::begin(wrong_rate_NOGD), std::end(wrong_rate_NOGD), 0.0);  
        double mean_NOGD =  sum / wrong_rate_NOGD.size(); 
        outFile << mean_time_NOGD << ',' << mean_NOGD << endl;
        //PROS-N-KONS
        sum = std::accumulate(std::begin(time_PROS), std::end(time_PROS), 0.0);  
        double mean_time_PROS =  sum / time_PROS.size(); 
        sum = std::accumulate(std::begin(wrong_rate_PROS), std::end(wrong_rate_PROS), 0.0);  
        double mean_PROS =  sum / wrong_rate_PROS.size();
        outFile << mean_time_PROS << ',' << mean_PROS << endl;
        //BOGD
        sum = std::accumulate(std::begin(time_BOGD), std::end(time_BOGD), 0.0);  
        double mean_time_BOGD =  sum / time_BOGD.size(); 
        sum = std::accumulate(std::begin(wrong_rate_BOGD), std::end(wrong_rate_BOGD), 0.0);  
        double mean_BOGD =  sum / wrong_rate_BOGD.size(); 
        outFile << mean_time_BOGD << ',' << mean_BOGD << endl;
    }
    outFile.close();
    
    
    return 0;
}


