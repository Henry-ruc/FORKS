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


    ifstream inFile("/data/ijcnn1/data.csv");
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
    ifstream inFile2("/data/ijcnn1/ID_ALL.csv");
    string lineStr2;
    vector<vector<float> > strArray2;
    while (getline(inFile2, lineStr2)) {
        //cout << lineStr << endl;
        stringstream ss(lineStr2);
        string str;
        vector<float> lineArray;
        while (getline(ss, str, ',')){

        float str_float;
        istringstream istr(str);
        istr >> str_float;
        lineArray.push_back(str_float);
        }
        strArray2.push_back(lineArray);
    }

    //baseline
    ofstream outFile;
    outFile.open("/ijcnn1.csv", ios::out); 

    std::vector<float> time_1nd;
    std::vector<float> wrong_rate_1nd;
    std::vector<float> time_2nd;
    std::vector<float> wrong_rate_2nd;
    std::vector<float> time_NOGD;
    std::vector<float> wrong_rate_NOGD;
    std::vector<float> time_PROS;
    std::vector<float> wrong_rate_PROS;
    std::vector<float> time_RBP;
    std::vector<float> wrong_rate_RBP;
    std::vector<float> time_BPAS;
    std::vector<float> wrong_rate_BPAS;
    std::vector<float> time_proj;
    std::vector<float> wrong_rate_proj;
    std::vector<float> time_BOGD;
    std::vector<float> wrong_rate_BOGD;
    for (int t = 0; t < 20; t++)
    {
        int N = 141691;
        int dim_x = 23;

        //SkeGD setting
        int B = 200;
        int sp = 150;
        int sm = 100;
        float eta = 0.2;
        int k = 0.1*sp;
        float sigma_SkeGD = 50;
        float sigma_SkeGD2nd = 50;
        int d = 1;
        float regularizer = 0.01;
        int rou = floor(0.3*(N));

        //else
        int Budget = 100; 
        float sigma = 8; 
        float C_BPAS = 1;
        int k_NOGD = floor(0.2*Budget);
        float lambda = 1.0/(N*N);
        float gamma = 10;


        SkeGD test_1nd(sp,sm,eta,k,B,rou,sigma_SkeGD,d);
        SkeGD2nd test_2nd(sp,sm,eta,k,B,rou,sigma_SkeGD2nd,d,regularizer);
        NOGD test_NOGD(eta,B,k,sigma);
        PROS_N_KONS test_PROS(C_BPAS,sigma);
        RBP test_RBP(Budget,sigma);
        BPAS test_BPAS(Budget,C_BPAS,sigma);
        Projectron test_proj(Budget,sigma);
        BOGD test_BOGD(eta,Budget,lambda,gamma,sigma);



        
        MatrixXf dataX = MatrixXf::Zero(N,dim_x-1);
        std::vector<float> dataY;
        for (int i = 0; i < N; i++)
        {
            int sample_index = 0;
            for (int j = 1; j < dim_x; j++)
            {
                sample_index = int(strArray2[t][i])-1;
                dataX(i,j-1) = strArray[sample_index][j];
            }
            dataY.push_back(strArray[sample_index][0]);
        }
        MatrixXf Y = Map<MatrixXf, Eigen::Unaligned>(dataY.data(), 1,dataY.size());

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

        system_clock::time_point beg_t_RBP = system_clock::now();
        test_RBP.RBP_learning(dataX,Y);
        system_clock::time_point end_t_RBP = system_clock::now();
        duration<double> diff_RBP = end_t_RBP - beg_t_RBP;
        time_RBP.push_back(diff_RBP.count());
        wrong_rate_RBP.push_back(float(test_RBP.wrong)/(N));

        system_clock::time_point beg_t_BPAS = system_clock::now();
        test_BPAS.BPAS_learning(dataX,Y);
        system_clock::time_point end_t_BPAS = system_clock::now();
        duration<double> diff_BPAS = end_t_BPAS - beg_t_BPAS;
        time_BPAS.push_back(diff_BPAS.count());
        wrong_rate_BPAS.push_back(float(test_BPAS.wrong)/(N));

        system_clock::time_point beg_t_proj = system_clock::now();
        test_proj.Projectron_learning(dataX,Y);
        system_clock::time_point end_t_proj = system_clock::now();
        duration<double> diff_proj = end_t_proj - beg_t_proj;
        time_proj.push_back(diff_proj.count());
        wrong_rate_proj.push_back(float(test_proj.wrong)/(N));

        system_clock::time_point beg_t_BOGD = system_clock::now();
        test_BOGD.BOGD_learning(dataX,Y);
        system_clock::time_point end_t_BOGD = system_clock::now();
        duration<double> diff_BOGD = end_t_BOGD - beg_t_BOGD;
        time_BOGD.push_back(diff_BOGD.count());
        wrong_rate_BOGD.push_back(float(test_BOGD.wrong)/(N));
    }
    //SkeGD
    double sum = std::accumulate(std::begin(time_1nd), std::end(time_1nd), 0.0);  
    double mean_time_1nd =  sum / time_1nd.size(); 
    sum = std::accumulate(std::begin(wrong_rate_1nd), std::end(wrong_rate_1nd), 0.0);  
    double mean_1nd =  sum / wrong_rate_1nd.size(); 
    double variance  = 0.0;
    for (uint16_t i = 0 ; i < wrong_rate_1nd.size() ; i++)
    {
        variance = variance + pow(wrong_rate_1nd[i]-mean_1nd,2);
    }
    variance = variance/wrong_rate_1nd.size();
    double standard_deviation = sqrt(variance); 
    outFile << mean_time_1nd << ',' << mean_1nd << ',' << standard_deviation << endl;
    //SkeGD2nd
    sum = std::accumulate(std::begin(time_2nd), std::end(time_2nd), 0.0);  
    double mean_time_2nd =  sum / time_2nd.size(); 
    sum = std::accumulate(std::begin(wrong_rate_2nd), std::end(wrong_rate_2nd), 0.0);  
    double mean_2nd =  sum / wrong_rate_2nd.size(); 
    variance  = 0.0;
    for (uint16_t i = 0 ; i < wrong_rate_2nd.size() ; i++)
    {
        variance = variance + pow(wrong_rate_2nd[i]-mean_2nd,2);
    }
    variance = variance/wrong_rate_2nd.size();
    standard_deviation = sqrt(variance); 
    outFile << mean_time_2nd << ',' << mean_2nd << ',' << standard_deviation << endl;
    //NOGD
    sum = std::accumulate(std::begin(time_NOGD), std::end(time_NOGD), 0.0);  
    double mean_time_NOGD =  sum / time_NOGD.size(); 
    sum = std::accumulate(std::begin(wrong_rate_NOGD), std::end(wrong_rate_NOGD), 0.0);  
    double mean_NOGD =  sum / wrong_rate_NOGD.size(); 
    variance  = 0.0;
    for (uint16_t i = 0 ; i < wrong_rate_NOGD.size() ; i++)
    {
        variance = variance + pow(wrong_rate_NOGD[i]-mean_NOGD,2);
    }
    variance = variance/wrong_rate_NOGD.size();
    standard_deviation = sqrt(variance); 
    outFile << mean_time_NOGD << ',' << mean_NOGD << ',' << standard_deviation << endl;
    //PROS-N-KONS
    sum = std::accumulate(std::begin(time_PROS), std::end(time_PROS), 0.0);  
    double mean_time_PROS =  sum / time_PROS.size(); 
    sum = std::accumulate(std::begin(wrong_rate_PROS), std::end(wrong_rate_PROS), 0.0);  
    double mean_PROS =  sum / wrong_rate_PROS.size(); 
    variance  = 0.0;
    for (uint16_t i = 0 ; i < wrong_rate_PROS.size() ; i++)
    {
        variance = variance + pow(wrong_rate_PROS[i]-mean_PROS,2);
    }
    variance = variance/wrong_rate_PROS.size();
    standard_deviation = sqrt(variance); 
    outFile << mean_time_PROS << ',' << mean_PROS << ',' << standard_deviation << endl;
    //RBP
    sum = std::accumulate(std::begin(time_RBP), std::end(time_RBP), 0.0);  
    double mean_time_RBP =  sum / time_RBP.size(); 
    sum = std::accumulate(std::begin(wrong_rate_RBP), std::end(wrong_rate_RBP), 0.0);  
    double mean_RBP =  sum / wrong_rate_RBP.size(); 
    variance  = 0.0;
    for (uint16_t i = 0 ; i < wrong_rate_RBP.size() ; i++)
    {
        variance = variance + pow(wrong_rate_RBP[i]-mean_RBP,2);
    }
    variance = variance/wrong_rate_RBP.size();
    standard_deviation = sqrt(variance); 
    outFile << mean_time_RBP << ',' << mean_RBP << ',' << standard_deviation << endl;
    //BPAS
    sum = std::accumulate(std::begin(time_BPAS), std::end(time_BPAS), 0.0);  
    double mean_time_BPAS =  sum / time_BPAS.size(); 
    sum = std::accumulate(std::begin(wrong_rate_BPAS), std::end(wrong_rate_BPAS), 0.0);  
    double mean_BPAS =  sum / wrong_rate_BPAS.size(); 
    variance  = 0.0;
    for (uint16_t i = 0 ; i < wrong_rate_BPAS.size() ; i++)
    {
        variance = variance + pow(wrong_rate_BPAS[i]-mean_BPAS,2);
    }
    variance = variance/wrong_rate_BPAS.size();
    standard_deviation = sqrt(variance); 
    outFile << mean_time_BPAS << ',' << mean_BPAS << ',' << standard_deviation << endl;
    //projectron
    sum = std::accumulate(std::begin(time_proj), std::end(time_proj), 0.0);  
    double mean_time_proj =  sum / time_proj.size(); 
    sum = std::accumulate(std::begin(wrong_rate_proj), std::end(wrong_rate_proj), 0.0);  
    double mean_proj =  sum / wrong_rate_proj.size(); 
    variance  = 0.0;
    for (uint16_t i = 0 ; i < wrong_rate_proj.size() ; i++)
    {
        variance = variance + pow(wrong_rate_proj[i]-mean_proj,2);
    }
    variance = variance/wrong_rate_proj.size();
    standard_deviation = sqrt(variance); 
    outFile << mean_time_proj << ',' << mean_proj << ',' << standard_deviation << endl;
    //BOGD
    sum = std::accumulate(std::begin(time_BOGD), std::end(time_BOGD), 0.0);  
    double mean_time_BOGD =  sum / time_BOGD.size(); 
    sum = std::accumulate(std::begin(wrong_rate_BOGD), std::end(wrong_rate_BOGD), 0.0);  
    double mean_BOGD =  sum / wrong_rate_BOGD.size(); 
    variance  = 0.0;
    for (uint16_t i = 0 ; i < wrong_rate_BOGD.size() ; i++)
    {
        variance = variance + pow(wrong_rate_BOGD[i]-mean_BOGD,2);
    }
    variance = variance/wrong_rate_BOGD.size();
    standard_deviation = sqrt(variance); 
    outFile << mean_time_BOGD << ',' << mean_BOGD << ',' << standard_deviation << endl;

    outFile.close();
    
    
    return 0;
}

