#include "CRRoption.h"
#include "binomial.h"
#include <iostream>
#include <cmath>
using namespace std;

int GetInputData(int& N, double& K) {
    cout << "Enter steps to expiry N: "; cin >> N;
    cout << "Enter strike price K: "; cin >> K;
    cout << endl;
    return 0;
}
