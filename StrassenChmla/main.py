import random
import time
import numpy as np
from matplotlib import pylab
import concurrent.futures as fur


def martixmmatrix(a, b):
    n = len(a)
    myres = []
    for i in range(n):
        midres = []
        for j in range(n):
            sum = 0
            for k in range(n):
                sum += a[i][k] * b[k][j]
            midres.append(sum)
        myres.append(midres)
    finRes = np.array(myres)
    return finRes


def splitMatrix(a, x0, y0, x1, y1):
    res = np.zeros(((x1-x0), (y1-y0)))
    n = len(res)
    for i in range(n):
        for j in range(n):
            res[i][j] = a[x0 + i][y0 + j]
    return res



def splitted(a, b, n):
    a11 = splitMatrix(a, 0, 0, n // 2, n // 2)
    a21 = splitMatrix(a, n // 2, 0, n, n // 2)
    a12 = splitMatrix(a, 0, n // 2, n // 2, n)
    a22 = splitMatrix(a, n // 2, n // 2, n, n)

    b11 = splitMatrix(b, 0, 0, n // 2, n // 2)
    b21 = splitMatrix(b, n // 2, 0, n, n // 2)
    b12 = splitMatrix(b, 0, n // 2, n // 2, n)
    b22 = splitMatrix(b, n // 2, n // 2, n, n)

    return a11, a21, a12, a22, b11, b12, b21, b22



def mulStrassen(a, b, nMin):
    n = len(a)
    if n <= nMin:
        return np.matmul(a, b)
    else:
        a11, a21, a12, a22, b11, b12, b21, b22 = splitted(a, b, n)

        with fur.ProcessPoolExecutor(max_workers=2) as executor:
            futP1 = executor.submit(mulStrassen, (a11 + a22), (b11 + b22), nMin)
            futP2 = executor.submit(mulStrassen, (a21 + a22), b11, nMin)
            futP3 = executor.submit(mulStrassen, a11, (b12 - b22), nMin)
            futP4 = executor.submit(mulStrassen, a22, (b21 - b11), nMin)
            futP5 = executor.submit(mulStrassen, (a11 + a12), b22, nMin)
            futP6 = executor.submit(mulStrassen, (a21 - a11), (b11 + b12), nMin)
            futP7 = executor.submit(mulStrassen, (a12 - a22), (b21 + b22), nMin)
            P1 = futP1.result()
            P2 = futP2.result()
            P3 = futP3.result()
            P4 = futP4.result()
            P5 = futP5.result()
            P6 = futP6.result()
            P7 = futP7.result()

        AB11 = P1 + P4 - P5 + P7
        AB12 = P3 + P5
        AB21 = P2 + P4
        AB22 = P1 - P2 + P3 + P6

        AB = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i < n // 2 and j < n // 2:
                    AB[i, j] = AB11[i, j]
                elif i < n // 2 and j >= n // 2:
                    AB[i, j] = AB12[i, j - n // 2]
                elif i >= n // 2 and j < n // 2:
                    AB[i, j] = AB21[i - n // 2, j]
                else:
                    AB[i, j] = AB22[i - n // 2, j - n // 2]

        return AB


def strassen(a, b, nMin):
    n = len(a)
    if n <= nMin:
        return np.matmul(a, b)
    else:
        a11, a21, a12, a22, b11, b12, b21, b22 = splitted(a, b, n)

        P1 = strassen((a11 + a22), (b11 + b22), nMin)
        P2 = strassen((a21 + a22), b11, nMin)
        P3 = strassen(a11, (b12 - b22), nMin)
        P4 = strassen(a22, (b21 - b11), nMin)
        P5 = strassen((a11 + a12), b22, nMin)
        P6 = strassen((a21 - a11), (b11 + b12), nMin)
        P7 = strassen((a12 - a22), (b21 + b22), nMin)


        res = np.zeros((n, n))
        for i in range(n //2):
            for j in range(n // 2):
                res[i][j] = P1[i][j] + P4[i][j] - P5[i][j] + P7[i][j]
        for i in range(n // 2):
            for j in range(n // 2):
                res[i + n // 2][j + n // 2] = P1[i][j] + P3[i][j] - P2[i][j] + P6[i][j]
        for i in range(n // 2):
            for j in range(n // 2):
                res[i + n // 2][j] = P2[i][j] + P4[i][j]
        for i in range(n // 2):
            for j in range(n // 2):
                res[i][j + n // 2] = P3[i][j] + P5[i][j]
        return res



timeStrMass = []
timeRegMass = []
timeMulStrMass = []
xMass = []
if __name__ == '__main__':
    n = 8
    for i in range(1, n):
        randA = [[random.random() for j in range(1 << i)] for k in range(1 << i)]
        randB = [[random.random() for j in range(1 << i)] for k in range(1 << i)]
        timeStr1 = time.time()
        strassenRes = strassen(randA, randB, 32)
        timeStr2 = time.time()
        timeStrMass.append(timeStr2 - timeStr1)

        timeReg1 = time.time()
        matrixMul = martixmmatrix(randA, randB)
        timeReg2 = time.time()
        timeRegMass.append(timeReg2 - timeReg1)

        timeMulStr1 = time.time()
        mulStr = mulStrassen(randA, randB, 32)
        timeMulStr2 = time.time()
        timeMulStrMass.append(timeMulStr2-timeMulStr1)

        normA = np.linalg.norm(strassenRes - matrixMul, ord='fro')

        # print("Strassen: ", strassenRes)
        # print("Reg: ", matrixMul)
        print("NORM: ", normA*100)
        xMass.append(i)
    pylab.plot(xMass, timeStrMass, label='Strassen')
    pylab.plot(xMass, timeRegMass, label='Regular')
    pylab.plot(xMass, timeMulStrMass, label='MulThread')
    pylab.savefig('strassen.png')

    pylab.legend(loc='upper left')
    pylab.show()

#
# a = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
# print(splitMatrix(a, 0, 0, 2, 2))