#ifndef NQS_H
#define NQS_H

#include "TFIM.cpp"

class NeuralQuantumState {
    
    public:
        
        //Network Parameters
        int epochs;                     // Training Steps
        int N;                          // Number of spins
        double lr;                      // Learning Rate
        double moment;                  // Momentum coefficient
        int MCS;                        // Monte Carlo Steps
        int eq;                         // Equilibration Steps

        int n_in;                       // Number of visible neurons
        int n_h;                        // Number of hidden neurons
        int n_P;                        // Number of Variational Parameters

        string optimization;            // Optimization method
        string objectiveFunc;           // Objective function
        string sign;                    // Network Sign (On/Off)

        VectorXd gradE;                 // Energy gradient
        VectorXd gradS;                 // Sign Gradient

        double Energy;                  // Energy

        double ExactEnergy;             // Exact Energy (ed-dmrg)
        double deltaEnergy;             // Relative error on the energy

        // Constructor
        NeuralQuantumState(MTRand & random, int N_,int MCS_, double lr_,
                int ep, PsiLayer& PL, HiddenLayer& HL,
                string opt_, string obj_, string sign_);

        // Core Functions
        void reset();
        void loadExactEnergy(string& model, int D); 

        VectorXd getEnergyGradient(MTRand & random,Hamiltonian& H, 
                                   PsiLayer& PL,HiddenLayer& HL); 

        
        void updateParameters(PsiLayer& PL, HiddenLayer& HL,
                              VectorXd& DELTA);

        void SGD_Plain(MTRand & random,Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL);
        void SGD_Momentum(MTRand & random,Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL);
        void SGD_AdaGrad(Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL);
        void SGD_AdaDelta(Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL);
        void SVRG(Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL);
        void SR(Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL, int e_);

        void optimizeNQS(MTRand & random,ofstream & file,
                   Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL);
 
        //// Utilities
        //void saveParameters(string& modelName);
        //void loadParameters(string& modelName);
        void printNetwork(PsiLayer& PL, HiddenLayer& HL);

};

#endif
