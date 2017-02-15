#ifndef AFHM_H
#define AFHM_H

#include <iostream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "MersenneTwister.h"
//#include <map>
//#include <complex>
#include "HiddenLayer.cpp"

using namespace std;
using namespace Eigen;

class Hamiltonian {
    
    public:
        
        // Hamiltonian Parameters
        int L;  // Linear size of the lattice
        int N;  // Number of physical spins
        int D;  // Dimension of the lattice
        int nL; // Number of interactions Links
        int NN; // Number of nearest neighbors
        string lattice;
      
        double Jxy;

        VectorXi NearestNeighbors;
        MatrixXi Coordinates;
       
        VectorXd spins; 
        
        // Constructor
        Hamiltonian(int D_, int L_, string latticeType);
        
        void initialize(MTRand & random,string sign);
 
        Vector2d getWaveFunction(PsiLayer& PL, HiddenLayer& HL);
        void flip(int& site1,int&site2);
        
        double getLocalEnergy(PsiLayer& PL, HiddenLayer& HL);
        void VQMC_sweep(MTRand & random, PsiLayer& PL, HiddenLayer & HL);

        void printLatticeInfos(); 
         
};

#endif
