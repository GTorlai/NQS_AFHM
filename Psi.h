#ifndef PSI_H
#define PSI_H

#include <iostream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "MersenneTwister.h"
//#include <map>
//#include <complex>

using namespace std;
using namespace Eigen;

class PsiLayer {
    
    public:
        
        //Network Parameters
        int n_in;
        double bound;
        
        MatrixXd Z;
        VectorXd c;

        // Constructor
        PsiLayer(MTRand & random, int nIN, double B);
        
        // Forward the input signal through the layer
        Vector2d getWF(const VectorXd & input); 

        //Parameters Saving and Loading
        void loadParameters(ifstream & file);
        void saveParameters(ofstream & file);
        
};

#endif
