#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <set>
#include <memory>
#include <thread>
#include <future>

#include "thread_pool.hpp"

#include "../lab_01/linalg.hpp"
#include "../lab_02/gplot.hpp"

static bool useThreads = false;
static std::unique_ptr<thread_pool> threadPool;

MyMatrix<float> getSubmatrix(MyMatrix<float> const &mat, size_t x0, size_t y0, size_t x1, size_t y1) {
    MyMatrix<float> res(x1 - x0, y1  - y0);
    for (size_t y = 0; y < res.size_y(); y++)
        for (size_t x = 0; x < res.size_x(); x++)
            res.at(x, y) = mat.at(x0 + x, y0 + y);
    return res;
}

MyMatrix<float> strassenRec(MyMatrix<float> const &a, MyMatrix<float> const &b) {
    size_t n = a.size_x();
    if (n <= 32)
        return a * b;

    auto A11 = getSubmatrix(a, 0, 0, n / 2, n / 2);
    auto B11 = getSubmatrix(b, 0, 0, n / 2, n / 2);

    auto A12 = getSubmatrix(a, n / 2, 0, n, n / 2);
    auto B12 = getSubmatrix(b, n / 2, 0, n, n / 2);

    auto A21 = getSubmatrix(a, 0, n / 2, n / 2, n);
    auto B21 = getSubmatrix(b, 0, n / 2, n / 2, n);

    auto A22 = getSubmatrix(a, n / 2, n / 2, n, n);
    auto B22 = getSubmatrix(b, n / 2, n / 2, n, n);

    MyMatrix<float> D, D1, D2, H1, H2, V1, V2;
    
    if (useThreads) {
        if (!threadPool) {
            auto futD  = std::async(std::launch::async, strassenRec, A11 + A22, B11 + B22);
            auto futD1 = std::async(std::launch::async, strassenRec, A12 - A22, B21 + B22);
            auto futD2 = std::async(std::launch::async, strassenRec, A21 - A11, B11 + B12);
            auto futH1 = std::async(std::launch::async, strassenRec, A11 + A12, B22);
            auto futH2 = std::async(std::launch::async, strassenRec, A21 + A22, B11);
            auto futV1 = std::async(std::launch::async, strassenRec, A22, B21 - B11);
            auto futV2 = std::async(std::launch::async, strassenRec, A11, B12 - B22);
            D  = futD.get();
            D1 = futD1.get();
            D2 = futD2.get();
            H1 = futH1.get();
            H2 = futH2.get();
            V1 = futV1.get();
            V2 = futV2.get();
        }
        else {
            assert(false); //! WARNING: Deadlock here
            auto futD =  threadPool->submit(strassenRec, A11 + A22, B11 + B22);
            auto futD1 = threadPool->submit(strassenRec, A12 - A22, B21 + B22);
            auto futD2 = threadPool->submit(strassenRec, A21 - A11, B11 + B12);
            auto futH1 = threadPool->submit(strassenRec, A11 + A12, B22);
            auto futH2 = threadPool->submit(strassenRec, A21 + A22, B11);
            auto futV1 = threadPool->submit(strassenRec, A22, B21 - B11);
            auto futV2 = threadPool->submit(strassenRec, A11, B12 - B22);
            D  = futD.get();
            D1 = futD1.get();
            D2 = futD2.get();
            H1 = futH1.get();
            H2 = futH2.get();
            V1 = futV1.get();
            V2 = futV2.get();
        }
    }
    else {
        D  = strassenRec(A11 + A22, B11 + B22);
        D1 = strassenRec(A12 - A22, B21 + B22);
        D2 = strassenRec(A21 - A11, B11 + B12);
        H1 = strassenRec(A11 + A12, B22);
        H2 = strassenRec(A21 + A22, B11);
        V1 = strassenRec(A22, B21 - B11);
        V2 = strassenRec(A11, B12 - B22);
    }

    MyMatrix<float> res(n, n);
    for (size_t x = 0; x < n / 2; x++) {
        for (size_t y = 0; y < n / 2; y++) {
            res.at(x, y) =
                D.at(x, y) + D1.at(x, y) + V1.at(x, y) - H1.at(x, y);
            res.at(x + n / 2, y + n / 2) =
                D.at(x, y) + D2.at(x, y) + V2.at(x, y) - H2.at(x, y);
            res.at(x + n / 2, y) =
                V2.at(x, y) + H1.at(x, y);
            res.at(x, y + n / 2) =
                V1.at(x, y) + H2.at(x, y);
        }
    }
    return res;
}


/*
 * Testing
 */

static MyVector<float> randomVector(size_t n, float maxv = 100.f) {
    MyVector<float> res(n);
    for (size_t i = 0; i < n; i++)
        res.at(i) = rand() / static_cast<float>(RAND_MAX) * maxv - maxv / 2.f;
    return res;
}

static MyMatrix<float> randomSymmMatrix(size_t n, float maxv = 100.f) {
    MyMatrix<float> res(n, n);
    for (auto &e : res.getData())
        e = 0.f;
    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) { // j
            res.at(i, j) = rand() / static_cast<float>(RAND_MAX) * maxv - maxv / 2.f;
        }
        res.at(j, j) = fabsf(res.at(j, j));
    }
    return res * res.transpose();
}

static MyMatrix<float> randomMatrix(size_t n, float maxv = 100.f) {
    MyMatrix<float> res(n, n);
    for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < n; i++)
            res.at(i, j) = rand() / static_cast<float>(RAND_MAX) * maxv - maxv / 2.f;
    return res;
}

static MyMatrix<float> getIdentity(size_t sz) {
    static std::unique_ptr<MyMatrix<float>> cached;
    if (cached && cached->size_x() == sz && cached->size_y() == sz)
        return *cached;

    cached = std::make_unique<MyMatrix<float>>(sz, sz);
    for (size_t i = 0; i < sz; i++)
        cached->at(i, i) = 1.f;
    return *cached;
}

static float matNormFrobenius(MyMatrix<float> const &mat) {
    float sum = 0.f;
    for (size_t y = 0; y < mat.size_y(); y++)
        for (size_t x = 0; x < mat.size_x(); x++)
            sum += mat.at(x, y);
    return sum;
}

static void makeTest(int n, bool threading) {
    useThreads = threading;
    std::chrono::steady_clock::time_point begin, end;
    auto ma = randomMatrix(n, 200.f);
    auto mb = randomMatrix(n, 200.f);
    auto exp = ma * mb;
    begin = std::chrono::steady_clock::now();
    auto got = strassenRec(ma, mb);
    end = std::chrono::steady_clock::now();
    printf("n=%d err=%.5f%% time=%ldms\n", n,
        matNormFrobenius(got - exp) / matNormFrobenius(exp) * 100.f,
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000L);
}


int main(int argc, char **argv) {
    int n;

    srand(0xFade18);
    makeTest(512, false);

    srand(0xFade18);
    makeTest(512, true);
    
    // threadPool = std::make_unique<thread_pool>(16);

    std::chrono::steady_clock::time_point begin, end;
    GnuPlot2d it_plot(2);
    it_plot.setName(0, "Single");
    it_plot.setName(1, "Multitheading");
    it_plot.setParam("logscale x");
    it_plot.setParam("xlabel 'matrix size'");
    it_plot.setParam("ylabel 'time, ms'");
    it_plot.setParam("xtics (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)");

    int max_n = 11;

    srand(0xFade18);
    useThreads = false;

    for (int i = 5; i < max_n; i++) {
        auto ma = randomMatrix(1 << i, 200.f);
        auto mb = randomMatrix(1 << i, 200.f);
        begin = std::chrono::steady_clock::now();
        auto got = strassenRec(ma, mb);
        end = std::chrono::steady_clock::now();
        it_plot.addPoint(0, 1 << i,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1'000L);
        printf("Single %d\n", i);
    }

    srand(0xFade18);
    useThreads = true;

    for (int i = 5; i < max_n; i++) {
        auto ma = randomMatrix(1 << i, 200.f);
        auto mb = randomMatrix(1 << i, 200.f);
        begin = std::chrono::steady_clock::now();
        auto got = strassenRec(ma, mb);
        end = std::chrono::steady_clock::now();
        it_plot.addPoint(1, 1 << i,
            std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1'000L);
        printf("Multi %d\n", i);
    }

    it_plot.writeGraph("time.png");

    return 0;
}