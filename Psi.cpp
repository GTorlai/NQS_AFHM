#include "Psi.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

PsiLayer::PsiLayer(MTRand& random, int nIN, double B) 
{
    
    n_in = int(nIN);
    bound = B;

    Z.setZero(n_in,2);
    c.setZero(2);

    for (int i=0; i<n_in; i++) {
        Z(i,0) = bound*(2.0 * random.rand() - 1.0);
        //Z(i,1) = bound*(2.0 * random.rand() - 1.0);
    }
}


//*****************************************************************************
// Activation 
//*****************************************************************************

Vector2d PsiLayer::getWF(const VectorXd & input) {
    
    Vector2d P;
    Vector2d activation;

    activation = Z.transpose() * input + c; 
    
    P(0) = activation(0);
 
    P(1) = 1.0;
    //P(1) = (1.0 - exp(-2.0*activation(1))) / (1.0 + exp(-2.0*activation(1)));
    
    return P;
}

//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void PsiLayer::loadParameters(ifstream & file) 
{
        
    for (int i=0; i<n_in; i++) {
        file >> Z(i,0);
        file >> Z(i,1);
    }
    file >> c(0);
    file >> c(1);
}


//*****************************************************************************
// Save the Network Parameters
//*****************************************************************************

void PsiLayer::saveParameters(ofstream & file) 
{

    for (int i=0; i<n_in; i++) {
        file << Z(i,0) << "  ";
        file << Z(i,1) << "  ";
    }
    file << endl << endl;
    
    file << c(0) << "  ";
    file << c(1);

    file << endl << endl;
}

