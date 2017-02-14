#include "HiddenLayer.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

HiddenLayer::HiddenLayer(MTRand& random, int nIn, int nH, double B) 
{
    
    n_in = int(nIn);
    n_h = int(nH);
    bound = B;
    
    W.setZero(n_in,n_h);
    b.setZero(n_h);

    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_h;i++) {
            W(k,i) = bound*(2.0 * random.rand() - 1.0);
        }
    }

    for (int i=0; i<n_h;i++) {
        b(i) = bound*(2.0 * random.rand() - 1.0);
    }

}

//*****************************************************************************
// Forward Pass 
//*****************************************************************************

VectorXd HiddenLayer::forward_pass(const VectorXd & input) {
    
    VectorXd h;
    VectorXd activation;

    activation = W.transpose() * input + b;
    h = sigmoid(activation); 
    
    return h;
}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void HiddenLayer::loadParameters(ifstream & file) 
{
        
    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_h;i++) {
            file >> W(k,i);
        }
    }
    
    for (int i=0; i<n_h;i++) {
        file >> b(i);
    }

}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void HiddenLayer::saveParameters(ofstream & file) 
{

    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_h;i++) {
            file << W(k,i) << " ";
        }
        file << endl;
    }
    
    file << endl << endl;

    for (int i=0; i<n_h;i++) {
        file << b(i) << " ";
    }
    file << endl << endl;

}

VectorXd HiddenLayer::sigmoid(const VectorXd & vec)                                        
{                                                                               
                                                                                    
    VectorXd X(vec.rows());                                    
                                                                                        
    for (int i=0; i< X.rows(); i++) {                                           
        X(i) = 1.0/(1.0+exp(-vec(i)));                               
    }                                             
                                                                                           
    return X;                                                                   
                                                                                                
}
