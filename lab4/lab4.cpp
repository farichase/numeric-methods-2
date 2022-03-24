#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <eigen3/Eigen/Dense>

using namespace std;

Eigen::MatrixXf to_eigen_m(vector<vector<float>> m) {
        Eigen::MatrixXf res(m.size(), m.size());
        for (size_t x = 0; x < m.size(); x++)
            for (size_t y = 0; y < m.size(); y++)
                res(x, y) = m[x][y];
        return res;
}
vector<vector<float>> randomMatrix(size_t n, float maxv = 20.f) {
    vector<vector<float>> res(n, vector<float>(n));
    float num = rand() / static_cast<float>(RAND_MAX) * maxv / 100;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i == j) res[i][j] = num;
            else res[i][j] = 0;
    return res;
}

vector<float> randomVector(int n, float maxv = 20.f) {
    vector<float> res(n);
    for (int i = 0; i < n; i++)
        res[i] = rand() / static_cast<float>(RAND_MAX) * maxv / 100;
    return res;
}
Eigen::VectorXf to_eigen_v(vector<float> v) {
    Eigen::VectorXf res(v.size());
    for (size_t x = 0; x < v.size(); x++)
        res(x) = v[x];
    return res;
}
Eigen::VectorXf oneParam(Eigen::MatrixXf Asymn, Eigen::VectorXf f, float e, float t, int &count, int n){
    Eigen::MatrixXf tAsymn = t * Asymn;
    vector<vector<float>> arr (n, vector<float> (n));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (i == j) arr[i][j] = 1;
            else arr[i][j] = 0;
        }
    }
    Eigen::MatrixXf E = to_eigen_m(arr);
    Eigen::MatrixXf G = E - tAsymn;
    auto g = f * t;
    vector<float> v(n);
    for (int i = 0; i < n; i++){
        v[i] = 0;
    }
    Eigen::VectorXf u = to_eigen_v(v);
    count = 0;
    while(true){
        count++;
        auto uC = u;
        u = G * u + g;
        if ((u - uC).norm() < e) break;
    } 
    return u;
}

int main(){
    
    size_t n = 100;
    size_t steps = 20;

    auto arr = randomMatrix(n, 50.f);

    Eigen::MatrixXf Asymn = to_eigen_m(arr);
    Eigen::VectorXf f = to_eigen_v(randomVector(n, 50.f));
    auto l = Asymn.eigenvalues();
    float eigen_max = l[0].real();
    float eigen_min = l[0].real();
    for (size_t i = 0; i < l.size(); i++) {
        if (eigen_max < l[i].real()) eigen_max = l[i].real();
        if (eigen_min > l[i].real()) eigen_min = l[i].real();
    }
    auto tOpt = 2 / (eigen_max + eigen_min);
    auto threshold = 2 / eigen_max;
    vector<float> tau;
    float step = threshold / steps;
    float val = 0.1;
    while (val < threshold){
        tau.push_back(val);
        val += step;
    }
    int count;
     Eigen::VectorXf u = oneParam(Asymn, f, 0.00001, tOpt, count, n);
     vector<int> counts;
     Eigen::MatrixXf check = Asymn * u;
     for (int i = 0; i < tau.size(); i++){
         oneParam(Asymn, f, 0.00001, tau[i], count, n);
         auto c = count;
         counts.push_back(c);
     }
     string datafile = "plot.dat";
     string scriptfile = "script";
     ofstream of(datafile);
     for (size_t i = 0; i < counts.size(); i++) {
             of << tau[i] << " " << counts[i] << std::endl;
     }
     of.flush();
     of.close();
     system(("gnuplot -c " + scriptfile).c_str());
     cout << "Optimal: " << tOpt << " " << count;
    return 0;
}