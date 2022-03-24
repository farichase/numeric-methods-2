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
vector<vector<float>> randomMatrix(size_t n) {
    vector<vector<float>> res(n, vector<float>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i == j) res[i][j] = rand() / static_cast<float>(RAND_MAX) * 10.f ;
    return res;
}

vector<float> randomVector(int n) {
    vector<float> res(n);
    for (int i = 0; i < n; i++)
        res[i] = rand() / static_cast<float>(RAND_MAX) * 10.f;
    return res;
}
Eigen::VectorXf to_eigen_v(vector<float> v) {
    Eigen::VectorXf res(v.size());
    for (size_t x = 0; x < v.size(); x++)
        res(x) = v[x];
    return res;
}
vector<vector<float>> tril(vector<vector<float>> arr){
    for (int i = 0; i < arr.size(); i++){
        for (int j = 0; j < arr[0].size(); j++){
            if (j > i) arr[i][j] = 0;
        }
    }
    return arr;
}
vector<vector<float>> toNormMat(Eigen::MatrixXf m, int n){
    vector<vector<float>> res(n, vector<float>(n));   
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            res[i][j] = m(i, j);
    return res;
}
vector<float> toNormVec(Eigen::VectorXf m, int n){
    vector<float> res(n);   
    for (size_t i = 0; i < n; i++)
        res[i] = m(i);
    return res;
}
vector<vector<float>> cholesky(vector<vector<float>> A, int n){
    vector<vector<float>> L (n, vector<float>(n));
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            L[i][j] = 0;
        }
    }
    L[0][0] = sqrt(A[0][0]);
    for (int i = 0; i < n; i++){
        L[i][0] = A[i][0] / L[0][0];
    }
    for (int i = 1; i < n; i++){
        float sum1 = 0;
        for (int j = 0; j < i; j++){
            sum1 += L[i][j] * L[i][j];
        }
        L[i][i] = sqrtf(A[i][i] - sum1);
        for (int j = i+1; j < n; j++){
            float sum2 = 0;
            for (int k = 0; k < i; k++){
                sum2 += L[i][k] * L[j][k];
            }
            L[j][i] = (A[j][i] - sum2) / L[i][i];
        }
    }
    return L;
}
vector<float> low_back_gauss(vector<vector<float>> L, vector<float> b, int n){
    vector<float> x (n);
    for (int i = 0; i < n; i++) {
        float sum0 = b[i];
        for (int j = 0; j < i; j++)
            sum0 -= L[i][j] * x[j];
        x[i] = sum0 / L[i][i];
    }
    return x;

}
vector<float> up_back_gauss(vector<vector<float>> L, vector<float> b, int n){
    vector<float> x (n);
    for (int i = n-1; i >-1; i--) {
        float sum0 = b[i];
        for (int j = n - 1; j >i; j--)
            sum0 -= L[i][j] * x[j];
        x[i] = sum0 / L[i][i];
    }
    return x;
}
pair<vector<vector<float>>, vector<vector<float>>> lu_decompose(vector<vector<float>> mat, int n) {
    vector<vector<float>> U(n, vector<float>(n));
    vector<vector<float>> L(n, vector<float>(n));

    for (int i = 0; i < n; i++)
        L[i][i] = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.f;
            if (i <= j) {
                for (int k = 0; k < i; k++)
                    sum += L[i][k] * U[k][j];
                U[i][j] = mat[i][j] - sum;
            }
            else {
                for (int k = 0; k < j; k++)
                    sum += L[i][k] * U[k][j];
                L[i][j] = (mat[i][j] - sum) / U[j][j];
            }
        }
    }

    return { L, U };
}
float lu_det(vector<vector<float>> L, vector<vector<float>> U) {
    float res = 1.f;
    for (size_t i = 0; i < L.size(); i++){
        res *= L[i][i] * U[i][i];
    }
    return res;
}
vector<vector<float>> lu_inverse(vector<vector<float>> L, vector<vector<float>> U) {
    int n = L.size();
    vector<vector<float>> res(n, vector<float>(n));
    
    vector<float> ecol (n);
    for (size_t c = 0; c < n; c++) {
        if (c != 0)
            ecol[c - 1] = 0.f;
        ecol[c] = 1.f;
        auto col = up_back_gauss(U, low_back_gauss(L, ecol, n), n);
        for (size_t i = 0; i < n; i++)
            res[i][c] = col[i];
    }

    return res;
}
vector<vector<float>> getIdentity(int n){
    vector<vector<float>> res(n, vector<float>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++){
            if (i == j) res[i][j] = 1;
            else res[i][j] = 0;
        }
    }
    return res;
}
float matnorm(vector<vector<float>> mat) {
    float max = 0.f;
    for (size_t y = 0; y < mat.size(); y++) {
        float cur = 0.f;
        for (size_t x = 0; x < mat.size(); x++)
            cur += fabsf(mat[y][x]);
        if (max < cur)
            max = cur;
    }
    return max;
}
int main(){
    size_t n = 100;
    srand(0xFade18);
    vector<vector<float>> arr = randomMatrix(n);
    vector<vector<float>> L0 = tril(arr);
    vector<vector<float>> A = toNormMat(to_eigen_m(L0) * to_eigen_m(L0).transpose(), n);
    vector<float> real = randomVector(n);
    vector<float> b = toNormVec(to_eigen_m(A) * to_eigen_v(real), n);

    vector<vector<float>> L = cholesky(A, n);
    vector<vector<float>> A0 = toNormMat(to_eigen_m(L) * to_eigen_m(L).transpose(), n);
    cout << "Cholesky result checking 1: " << abs(to_eigen_m(A).norm() - to_eigen_m(A0).norm()) << "\n";

    vector<float> y = low_back_gauss(L, b, n);
    vector<vector<float>> LT = toNormMat(to_eigen_m(L).transpose(), n);
    vector<float> x = up_back_gauss(LT, y, n);
    cout << "Cholesky result checking 2: " << (to_eigen_v(x) - to_eigen_v(real)).norm() / to_eigen_v(real).norm() << "\n";
    
    n = 15;
    vector<vector<float>> A2 = randomMatrix(n);
    vector<float> real2 = randomVector(n);
    vector<float> b2 = toNormVec(to_eigen_m(A2) * to_eigen_v(real2), n);
    pair<vector<vector<float>>, vector<vector<float>>> l2u2 = lu_decompose(A2, n);
    vector<vector<float>> l2 = l2u2.first;
    vector<vector<float>> u2 = l2u2.second;
    cout <<  "LU result checking 1: " << (to_eigen_m(l2) * to_eigen_m(u2) - to_eigen_m(A2)).norm() << "\n";

    vector<float> y2 = low_back_gauss(l2, b2, n);
    vector<float> x2 = up_back_gauss(u2, y2, n);
    cout << "LU result checking 2: " << (to_eigen_v(x2) - to_eigen_v(real2)).norm() / to_eigen_v(real2).norm() << "\n";

    float det = lu_det(l2, u2);
    float ldet = to_eigen_m(A2).determinant();
    cout << "LU result checking 3: " << det - ldet<< "\n";

    auto inv = lu_inverse(l2, u2);
    Eigen::MatrixXf check = to_eigen_m(A2) * to_eigen_m(inv);
    auto m = toNormMat(check - to_eigen_m(getIdentity(n)), n);
    cout << "LU result checking 4: " << matnorm(m) << "\n";

    return 0;
}