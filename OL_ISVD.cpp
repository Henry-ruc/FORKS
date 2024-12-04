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


    ifstream inFile("/data/codrna/data.csv");
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
    ifstream inFile2("/data/codrna/ID_ALL.csv");
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
    


    ofstream outFile;
    outFile.open("/OL_ISVD.csv", ios::out); 
    float rho_list[8] = {0.5,0.3,0.1,0.05,0.01,0.005,0.001,0.0005};

    for (int rho_loop = 0; rho_loop < 8; rho_loop++)
    {
        std::vector<float> time_1nd;
        std::vector<float> wrong_rate_1nd;
        std::vector<float> time_2nd;
        std::vector<float> wrong_rate_2nd;
        std::vector<float> time_NOGD;
        std::vector<float> wrong_rate_ori_2nd;
        std::vector<float> time_ori_2nd;
        for (int t = 0; t < 20; t++)
        {
            int N = 59535;
            int dim_x = 9;

            //SkeGD setting
            int B = 200;
            int sp = 150;
            int sm = 100;
            float eta = 0.2;
            int k = 0.1*sp;
            float sigma_SkeGD = 50;
            float sigma_SkeGD2nd = 100;
            int d = 1;
            float regularizer = 0.01;
            int rou = floor(rho_list[rho_loop]*(N));


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


            SkeGD2nd test_2nd(sp,sm,eta,k,B,rou,sigma_SkeGD,d,regularizer);
            SkeGD2nd_origin test_origin_2nd(sp,sm,eta,k,B,rou,sigma_SkeGD,d,regularizer);

            system_clock::time_point beg_t_2nd = system_clock::now();
            test_2nd.SkeGD_learning(dataX,Y);
            system_clock::time_point end_t_2nd = system_clock::now();
            duration<double> diff_2nd = end_t_2nd - beg_t_2nd;
            time_2nd.push_back(diff_2nd.count());
            wrong_rate_2nd.push_back(float(test_2nd.wrong)/(N));

            system_clock::time_point beg_t_ori_2nd = system_clock::now();
            test_origin_2nd.SkeGD_learning(dataX,Y);
            system_clock::time_point end_t_ori_2nd = system_clock::now();
            duration<double> diff_ori_2nd = end_t_ori_2nd - beg_t_ori_2nd;
            time_ori_2nd.push_back(diff_ori_2nd.count());
            wrong_rate_ori_2nd.push_back(float(test_origin_2nd.wrong)/(N));

        }
        outFile << rho_list[rho_loop] << endl;

        //SkeGD2nd
        double sum = std::accumulate(std::begin(time_2nd), std::end(time_2nd), 0.0);  
        double mean_time_2nd =  sum / time_2nd.size(); 
        sum = std::accumulate(std::begin(wrong_rate_2nd), std::end(wrong_rate_2nd), 0.0);  
        double mean_2nd =  sum / wrong_rate_2nd.size(); 
        double variance  = 0.0;
        for (uint16_t i = 0 ; i < wrong_rate_2nd.size() ; i++)
        {
            variance = variance + pow(wrong_rate_2nd[i]-mean_2nd,2);
        }
        variance = variance/wrong_rate_2nd.size();
        double standard_deviation = sqrt(variance); 
        outFile << mean_time_2nd << ',' << mean_2nd << ',' << standard_deviation << endl;
        //SkeGD2nd_no_ISVD
        sum = std::accumulate(std::begin(time_ori_2nd), std::end(time_ori_2nd), 0.0);  
        double mean_time_ori_2nd =  sum / time_ori_2nd.size(); 
        sum = std::accumulate(std::begin(wrong_rate_ori_2nd), std::end(wrong_rate_ori_2nd), 0.0);  
        double mean_ori_2nd =  sum / wrong_rate_ori_2nd.size(); 
        variance  = 0.0;
        for (uint16_t i = 0 ; i < wrong_rate_ori_2nd.size() ; i++)
        {
            variance = variance + pow(wrong_rate_ori_2nd[i]-mean_ori_2nd,2);
        }
        variance = variance/wrong_rate_ori_2nd.size();
        standard_deviation = sqrt(variance); 
        outFile << mean_time_ori_2nd << ',' << mean_ori_2nd << ',' << standard_deviation << endl;
    }


    outFile.close();
    
    
    return 0;
}

