#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <eigen3/Eigen/Dense>

using namespace std;

void elimination(vector<vector<float>> &A, vector<float> &f, int i){
    for (int j = i + 1; j < f.size(); j++){
        float c = A[i][j] / A[i][i];
        A[i][j] = 0.f;
        for (int k = i + 1; k < f.size(); k++){
            A[k][j] -= c * A[k][i];
        }
        f[j] -= c * f[i];
    }
}

void backward(vector<vector<float>> &A, vector<float> &f) {
    for (int i = f.size() - 1; i > -1; i--){
        f[i] = f[i] / A[i][i];
        for (int j = i - 1; j > -1; j--) {
            f[j] -= A[i][j] * f[i];
        }
    }
}
vector<float> gauss(vector<vector<float>> A, vector<float> f) {
    for (int i = 0; i < f.size(); i++){
        elimination(A, f, i);
    }
    backward(A, f);
    return f;
}

vector<float> colGauss(vector<vector<float>> A, vector<float> f) {
    for (int i = 0; i < f.size(); i++) {
        int iMax = i;
        float valMax = abs(A[i][i]);
        for (int j = i + 1; j < f.size(); j++){
            if (valMax < abs(A[i][j])){
                valMax = abs(A[i][j]);
                iMax = j;
            }
        }
        if (iMax != i)
        {
            for (int j = i; j < f.size(); j++)
            {
                float t = A[j][i];
                A[j][i] = A[j][iMax];
                A[j][iMax] = t;
            }
            float t = f[i];
            f[i] = f[iMax];
            f[iMax] = t;
        }
        elimination(A, f, i);
    }
    backward(A, f);
    return f;
}
vector<float> rowGauss(vector<vector<float>> A, vector<float> f) {
    vector<int> xBack(f.size());
    for (int i = 0; i < xBack.size(); i++) {
        xBack[i] = i;
    }
    for (int i = 0; i < xBack.size(); i++) {
        int jMax = i;
        float valmax = abs(A[i][i]);
        for (int j = i + 1; j < f.size(); j++){
            if (valmax < abs(A[j][i])){
                valmax = abs(A[j][i]);
                jMax = j;
            }
        }
        if (jMax != i) {
            for (int j = 0; j < f.size(); j++){
                float t = A[i][j];
                A[i][j] = A[jMax][j];
                A[jMax][j] = t;
            }
            int t = xBack[i];
            xBack[i] = xBack[jMax];
            xBack[jMax] = t;
        }
        elimination(A, f, i);
    }
    backward(A, f);
    vector<float> res(f.size());
    for (int i = 0; i < res.size(); i++) {
        res[xBack[i]] = f[i];
    }
    return res;
}

vector<float> dualGauss(vector<vector<float>> A, vector<float> f) {
    vector<int> xBack(f.size());
    for (int i = 0; i < xBack.size(); i++) {
        xBack[i] = i;
    }
    for (int i = 0; i < xBack.size(); i++) {
        int maxRow = i;
        float valRow = abs(A[i][i]);
        for (int j = i + 1; j < f.size(); j++){
            if (valRow < abs(A[i][j])){
                valRow = abs(A[i][j]);
                maxRow = j;
            }
        }
        int maxCol = i;
        float valCol = abs(A[i][i]);
        for (int j = i + 1; j < f.size(); j++){
            if (valRow < abs(A[j][i])){
                valRow = abs(A[j][i]);
                maxRow = j;
            }
        }
        if (valCol > valRow) {
            if (maxCol != i) {
                for (int j = 0; j < f.size(); j++){
                    float t = A[i][j];
                    A[i][j] = A[maxCol][j];
                    A[maxCol][j] = t;
                }
                int t = xBack[i];
                xBack[i] = xBack[maxCol];
                xBack[maxCol] = t;
            }
        }
        else if (maxRow != i) {
            for (int j = i; j < f.size(); j++){
                float t = A[j][i];
                A[j][i] = A[j][maxRow];
                A[j][maxRow] = t;
            }
            float t = f[i];
            f[i] = f[maxRow];
            f[maxRow] = t;
        }
        elimination(A, f, i);
    }
    backward(A, f);
    vector<float> res(f.size());
    for (int i = 0; i < res.size(); i++) {
        res[xBack[i]] = f[i];
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

vector<float> randomVector(int n, float maxv = 100.f) {
    vector<float> res(n);
    for (int i = 0; i < n; i++)
        res.at(i) = rand() / static_cast<float>(RAND_MAX) * maxv;
    return res;
}
vector<vector<float>> randomMatrix(size_t n, float maxv) {
    vector<vector<float>> res(n, vector<float>(n));   
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            // if (i != j) res[i][j] = rand() / static_cast<float>(RAND_MAX) * maxv;
            // else res[i][j] = 600000.f;
            res[i][j] = -100 + (rand() / ( RAND_MAX / ( maxv +100) ) ) ; 
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
Eigen::VectorXf to_eigen_v(vector<float> v) {
    Eigen::VectorXf res(v.size());
    for (size_t x = 0; x < v.size(); x++)
        res(x) = v[x];
    return res;
}
Eigen::MatrixXf to_eigen_m(vector<vector<float>> m) {
        Eigen::MatrixXf res(m.size(), m.size());
        for (size_t x = 0; x < m.size(); x++)
            for (size_t y = 0; y < m.size(); y++)
                res(x, y) = m[x][y];
        return res;
    }
float checkResult(vector<vector<float>> &A, vector<float> &f, vector<float> &res){
    // Eigen::VectorXf check = to_eigen_m(A) * to_eigen_v(res);
    // return (check - to_eigen_v(f)).norm();


    auto check = matrixVectorMultiplication(A, res);
    return norm(subtraction(check, f));
}
void test(vector<vector<float>> A, vector<float> f, vector<float> realX){
    printf("%s\n", diagCheck(A) ? "Diagonal dominant" : "Normal");

    auto gs = gauss(A, f);
    auto col = colGauss(A, f);
    auto row = rowGauss(A, f);
    auto dual = dualGauss(A, f);

    printf("Gauss: %e\n", checkResult(A, f, gs));
    printf("Col: %e\n", checkResult(A, f, col));
    printf("Row: %e\n", checkResult(A, f, row));
    printf("Dual: %e\n", checkResult(A, f, dual));
    printf("------------------\n");

    printf("Gauss rel Dual: %e\n", norm(subtraction(gs, dual)) / norm(dual) * 100);
    printf("Col rel Dual: %e\n", norm(subtraction(col, dual)) / norm(dual) * 100);
    printf("Row  rel Dual: %e\n", norm(subtraction(row, dual)) / norm(dual) * 100);

    printf("Gauss rel Real: %e\n", norm(subtraction(gs, realX)) / norm(realX) * 100);
    printf("Col rel Real: %e\n", norm(subtraction(col, realX)) / norm(realX) * 100);
    printf("Row  rel Real: %e\n", norm(subtraction(row, realX)) / norm(realX) * 100);
    printf("Dual rel Real: %e\n", norm(subtraction(gs, realX)) / norm(realX) * 100);

}
float norm2(vector<vector<float>> A) {
    float max = 0.f;
    for (int i = 0; i < A.size(); i++){;
        float sum = 0.f;
        for (int j = 0; j < A[0].size(); j++){
            sum += abs(A[i][j]);
        }
        if (sum > max) max = sum;
    }
    return max;
}
vector<vector<float>> toNormMat(Eigen::MatrixXf m, int n){
    vector<vector<float>> res(n, vector<float>(n));   
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            res[i][j] = m(i, j);
    return res;
}
float conditionNumber(vector<vector<float>> A){
    Eigen::MatrixXf inverseA = to_eigen_m(A).inverse();
    return norm2(A) * norm2(toNormMat(inverseA, A.size()));
}
int main(){
    srand(0xFace);
    // size_t n = 3;

    // auto A = randomMatrix(n, 300.);
    // auto realX = randomVector(n, 300.);
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
    // // test(A, f, realX);
    // cout << "Condition number of A " << conditionNumber(A);
    srand(0xDead0); // 0xDead0f

    vector<float> pointsX;
    vector<vector<float>> pointsY (4);
    vector<float> condY;
    
    string datafile = "plot.dat";
    string datafile2 = "plot2.dat";
    string scriptfile = "script";
    string scriptfile2 = "script2";
    for (int i = 2; i < 101; i++) {
        auto A = randomMatrix(i, 50.f);
        auto f = randomVector(i, 50.f);

        if (diagCheck(A)) printf("Diagonal dominant\n");

        auto gs = gauss(A, f);
        auto col = colGauss(A, f);
        auto row = rowGauss(A, f);
        auto dual = dualGauss(A, f);
        auto cond = conditionNumber(A)/1000;

        float gsP = norm(subtraction(gs, dual)) / norm(dual) * 100;
        float colP = norm(subtraction(col, dual)) / norm(dual) * 100;
        float rowP = norm(subtraction(row, dual)) / norm(dual) * 100;

        pointsX.push_back(i);
        condY.push_back(cond);
        pointsY[0].push_back(gsP);
        pointsY[1].push_back(colP);
        pointsY[2].push_back(rowP);
        pointsY[3].push_back(cond);
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
        for (size_t i = 0; i < pointsX.size(); i++) {
            of2 << pointsX[i] << " " << condY[i] << std::endl;
        }
        of2 << std::endl << std::endl;
    
    of2.flush();
    of2.close();
        
    system(("gnuplot -c " + scriptfile).c_str());
    system(("gnuplot -c " + scriptfile2).c_str());
    return 0;
}