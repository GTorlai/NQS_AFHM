#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include "Psi.cpp"

using namespace std;

class HiddenLayer {
    
    public:
        
        //Network Parameters
        int n_in;
        int n_h;
        double bound;

        MatrixXd W;
        VectorXd b;

        // Constructor
        HiddenLayer(MTRand & random, int nIN, int nH, double B);
         
        // Functions
        VectorXd forward_pass(const VectorXd & input);
       
        VectorXd sigmoid(const VectorXd & vec); 
        
        void loadParameters(ifstream & file);
        void saveParameters(ofstream & file);
        
};

#endif
