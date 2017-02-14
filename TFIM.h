#ifndef TFIM_H
#define TFIM_H

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
       
        double h;
        double J;
        
        VectorXi NearestNeighbors;
        MatrixXi Coordinates;
       
        VectorXd spins; 
        
        // Constructor
        Hamiltonian(int D_, int L_, string latticeType);
        
        void initialize(MTRand & random,string interaction,double h_,string sign);
 
        Vector2d getWaveFunction(PsiLayer& PL, HiddenLayer& HL);
        void flip(int& site);
        
        double getLocalEnergy(PsiLayer& PL, HiddenLayer& HL);
        void VQMC_sweep(MTRand & random, PsiLayer& PL, HiddenLayer & HL);

        void printLatticeInfos(); 
         
};



//class TFIM: public HamBase {
//
//    public:
//        
//        TFIM(string& interaction,double& h_,string& sign); 
//        double h;
//        double J;
//
//        //void initializeHamiltonian(string& interaction,double& h_,string& sign);
//
//        double getLocalEnergy(PsiLayer& PL, HiddenLayer& HL);
//        void VQMC_sweep(MTRand & random, PsiLayer& PL, HiddenLayer & HL);
//
//
//};
//
//
//class AFHM: public HamBase {
//
//    public:
//         
//        void initializeHamiltonian(string& interaction,string& sign);
//
//        double getLocalEnergy(PsiLayer& PL, HiddenLayer& HL);
//        void VQMC_sweep(MTRand & random, PsiLayer& PL, HiddenLayer & HL);
//
//
//};


#endif
