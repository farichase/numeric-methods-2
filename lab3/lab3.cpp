#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

using namespace std;
int iterations;

vector<float> randomVector(int n, float maxv = 100.f) {
    vector<float> res(n);
    for (int i = 0; i < n; i++)
        res[i] = rand() / static_cast<float>(RAND_MAX) * 10;
    return res;
}
vector<vector<float>> randomMatrix(size_t n, float maxv) {
    vector<vector<float>> res(n, vector<float>(n));   
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            if (i != j) res[i][j] = -10 + (rand() / ( RAND_MAX / ( maxv +10) ) ) ; 
            else res[i][j] = 600.f;
    return res;
}
vector<float> matrixVectorMultiplication(vector<vector<float>> m, vector<float> v){
    vector<float> res;
    if (m[0].size() == v.size()) {
        for (int i = 0; i < m.size(); i++)
        {
            float sum = 0;

            for (int j = 0; j < v.size(); j++)
            {
                sum += m[i][j] * v[j];
            }
            res.push_back(sum);
        }
    }
    return res;
}

bool diagCheck(vector<vector<float>> &A){
    float total = 0.f;
    for (int i = 0; i < A.size(); i++){
        for (int j = 0; j < A.size(); j++){
            if (i != j) {
                total += abs(A[i][j]);
            }
        }
    }
    bool strict = false;
    for (int i = 0; i < A.size(); i++){
        if (abs(A[i][i]) < total) {
            return false;
        }
        if (!strict && abs(A[i][i]) > total) strict = true;
    }
    return strict;
}
vector<float> subtraction(vector<float> v1, vector<float> v2){
    vector<float> res(v1.size());
    for (int i = 0; i < res.size(); i++){
        res[i] = v1[i] - v2[i];
    }
    return res;
}
float norm(vector<float> v){
    float sum = 0.f;
    for (int i = 0; i < v.size(); i++){
        sum += v[i] * v[i];
    }
    return sqrtf(sum);
}
float checkResult(vector<vector<float>> A, vector<float> f, vector<float> x){

    auto check = matrixVectorMultiplication(A, x);
    auto sub = subtraction(check, f);
    // for (int i = 0; i < f.size(); i++){
    //     cout << check[i] << " " << f[i] << "\n";
    // }
    return norm(subtraction(check, f)) / norm(f);
}
vector<float> stepS(vector<vector<float>> A, vector<float> f, vector<float> x, int n){
    auto next = f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            next[i] -= A[i][j] * next[j];
        }
        for (int j = i + 1; j < n; j++) {
            next[i] -= A[i][j] * x[j];
        }
        next[i] /= A[i][i];
    }
    return next;
}
vector<float> stepJ(vector<vector<float>> A, vector<float> f, vector<float> x, int n){
    auto next = f;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;
            next[i] -= A[i][j] * x[j];
        }
        next[i] /= A[i][i];
    }

    return next;
}
vector<float> seidel(vector<vector<float>> A, vector<float> f, int n){
    auto eps = 0.001;
    
    vector<float> curApprox(n);
    vector<float> nextApprox(n);
    for (size_t i = 0; i < n; i++)
        nextApprox[i] = 0.f;
    int iter = 0;
    do {
        curApprox = nextApprox;
        nextApprox = stepS(A, f, curApprox, n);
        iter++;
    } while (norm(subtraction(nextApprox ,curApprox)) > 1e-3);
    iterations = iter;
    // cout << "Итераций по методу Зейделя " << iter << "\n";

    return nextApprox;

}
vector<float> jacobi(vector<vector<float>> A, vector<float> f, int n){
    vector<float> curApprox(n);
    vector<float> nextApprox(n);
    for (size_t i = 0; i < n; i++)
        nextApprox[i] = 0.f;
    int iter = 0;
    do {
        curApprox = nextApprox;
        nextApprox = stepJ(A, f, curApprox, n);
        iter++;
    } while (norm(subtraction(nextApprox ,curApprox)) > 1e-5);
    iterations = iter;
    // cout << "Итераций по методу Якоби " << iter << "\n";
    return nextApprox;
}
int main()
{   
    // int n = 10;

    // auto A = randomMatrix(n, 10.);
    // auto realX = randomVector(n, 10.);
    // auto f = matrixVectorMultiplication(A, realX);
    
    // ofstream mfile("matrix.txt");
    // for (int i = 0; i < n; i++){
    //     for (int j = 0; j < n; j++){
    //         mfile << A[i][j] << " ";
    //     }
    //     mfile << "\n";
    // }
    // mfile.close();
    // ofstream vfile("vector.txt");
    // for (int i = 0; i < n; i++){
    //     vfile << realX[i] << " ";
    // }
    // vfile.close();
    // vector<float> res = seidel(A, f, n);
    // cout << checkResult(A, f, res) << "\n";
    // vector<float> res2 = jacobi(A, f, n);
	// cout << checkResult(A, f, res2);

    string datafile = "plot.dat";
    string datafile2 = "plot2.dat";
    string scriptfile = "script";
    string scriptfile2 = "script2";
    vector<float> pointsX;
    vector<vector<float>> pointsY(2);
    vector<vector<float>> pointsY2(2);
    for (int i = 2; i < 200; i+=5) {
        auto A = randomMatrix(i, 10.f);
        auto f = randomVector(i, 10.f);

        vector<float> res = seidel(A, f, i);

        pointsY2[0].push_back(iterations);
        // cout << checkResult(A, f, res) << "\n";
        vector<float> res2 = jacobi(A, f, i);
        pointsY2[1].push_back(iterations);
        // cout << checkResult(A, f, res2);
        pointsX.push_back(i);
        pointsY[0].push_back(checkResult(A, f, res));
        pointsY[1].push_back(checkResult(A, f, res2));
    }
    ofstream of(datafile);
    for (size_t si = 0; si < pointsY.size(); si++) {
        for (size_t i = 0; i < pointsY[0].size(); i++) {
            of << pointsX[i] << " " << pointsY[si][i] << std::endl;
        }
        of << std::endl << std::endl;
    }
    of.flush();
    of.close();

    ofstream of2(datafile2);
    for (size_t si = 0; si < pointsY.size(); si++) {
        for (size_t i = 0; i < pointsY[0].size(); i++) {
            of2 << pointsX[i] << " " << pointsY2[si][i] << std::endl;
        }
        of2 << std::endl << std::endl;
    }
    of2.flush();
    of2.close();
        
    system(("gnuplot -c " + scriptfile).c_str());
    system(("gnuplot -c " + scriptfile2).c_str());
    
    return 0;
}