#include "NQS.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iomanip>
//*****************************************************************************
// Constructor 
//*****************************************************************************

NeuralQuantumState::NeuralQuantumState(
        MTRand& random, 
        int N_,
        int MCS_, 
        double lr_, 
        int ep_,
        PsiLayer& PL, 
        HiddenLayer& HL,
        string opt_,
        string obj_,
        string sign_) 
{
    
    lr = lr_;
    MCS = int(MCS_);
    N = int(N_);
    epochs = int(ep_);
    eq = int(MCS/10);
    moment = 0.9;
    ExactEnergy = 0;
    optimization = opt_;
    objectiveFunc = obj_;
    sign = sign_;


    n_in = N;
    n_h  = HL.n_h;
    
    if (sign.compare("on") == 0) 
        n_P  = int(n_h*(n_in+3));
    else
        n_P  = int(n_h*(n_in+2));

    gradE.setZero(n_P);
        
}


//*****************************************************************************
// Reset the energy and the gradients 
//*****************************************************************************

void NeuralQuantumState::reset() {

    Energy = 0.0; 
    gradE.setZero(n_P);
    gradS.setZero(n_P); 
}


//*****************************************************************************
// Compute the gradient of the energy w.r.t. variational parameters 
//*****************************************************************************

void NeuralQuantumState::loadExactEnergy(string& model, int D)
{

    string fileName;
    fileName = "data/ed-dmrg/energy_dmrg_";
    fileName += boost::str(boost::format("%d") % D);
    fileName += "d";
    fileName += model;
    fileName += ".dat";

    ifstream fin(fileName);

    double e;
    double temp; 
    
        
    for (int i=0; i<N-3; i++) {
        fin >> temp;
        
    }
    fin >> e;
    ExactEnergy = e; 

}


//*****************************************************************************
// Compute the gradient of the energy w.r.t. variational parameters 
//*****************************************************************************

VectorXd NeuralQuantumState::getEnergyGradient(MTRand & random,Hamiltonian& H, 
                                               PsiLayer& PL, HiddenLayer& HL) 
{

    VectorXd gE;
    VectorXd O;
    VectorXd Oe;
    
    gE.setZero(n_P);
    O.setZero(n_P);
    Oe.setZero(n_P);
     
    VectorXd h;
    double eL;
    double E;
    
    int c;
 
    // Equilibration Stage
    //for (int n=0; n<eq; n++) {
    //    
    //    H.VQMC_sweep(random,PL,HL); 
    //        
    //} 
    
    // Compute statistics for the energy gradient
    for (int n=0; n<MCS; n++) {
        
        H.VQMC_sweep(random,PL,HL); 
        
        // Compute Hidden Layer state
        h = HL.forward_pass(H.spins);

        // Compute Local Energy
        eL = H.getLocalEnergy(PL,HL);
        c = 0;
        
        // Accumulate statistics of O on W
        for (int k=0; k<n_in; k++) {
            for (int i=0; i<n_h; i++) {
                O(c)  += h(i)*(1.0-h(i))*PL.Z(i,0)*H.spins(k);
                Oe(c) += h(i)*(1.0-h(i))*PL.Z(i,0)*H.spins(k)*eL;
                c++;
            }
        }
        
        // Accumulate statistics of O on b
        for (int i=0; i<n_h; i++) {
            O(c)  += h(i)*(1.0-h(i))*PL.Z(i,0);
            Oe(c) += h(i)*(1.0-h(i))*PL.Z(i,0)*eL;
            c++;
        }
     
        // Accumulate statistics of O on Z
        for (int i=0;i<n_h; i++) {
            O(c)  += h(i);
            Oe(c) += h(i)*eL;
            c++;
        }
        
        //cout << eL << endl << endl;
        // Accumulate statistics for energy    
        E += eL;
    }
     
    Energy = E/double(N*MCS);
    
    deltaEnergy = abs((Energy-ExactEnergy) / ExactEnergy);

    // Compute the gradient of the energy
    for (int c=0; c<n_P; c++) {
        
        gE(c) = 2*(Oe(c)/(double(MCS))-O(c)*E/(double(MCS*MCS)));
    }

    return gE;
}


//*****************************************************************************
// Update all the NQS variational parameters 
//*****************************************************************************

void NeuralQuantumState::updateParameters(PsiLayer& PL, HiddenLayer& HL,
                                          VectorXd& DELTA) {
        
    int c = 0;

    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_h; i++) {
            HL.W(k,i) += DELTA(c); 
            c++;
        }
    }
    
    for (int i=0; i<n_h; i++) {
        HL.b(i) += DELTA(c);
        c++;
    }

    for (int i=0; i<n_h; i++) {
        PL.Z(i) += DELTA(c);
        c++;
    }        
}


void NeuralQuantumState::SGD_Plain(MTRand & random, Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL)
{

    VectorXd gE(n_P);
    VectorXd delta;
    
    delta.setZero(n_P); 
    
    for (int e=0; e<epochs; e++) {
        
        gE = getEnergyGradient(random,H,PL,HL);
        delta = -lr*gE;
        updateParameters(PL,HL,delta);
        //cout << HL.W << endl;
        //cout << HL.W(0,0) << endl;
        cout << "Epoch # " << e << "\tGS Energy =  " << Energy;
        cout << "Error = " << setprecision(8) << deltaEnergy << endl;
    }
}


void NeuralQuantumState::SGD_Momentum(MTRand & random, Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL)
{

    VectorXd gE(n_P);
    VectorXd delta;
    
    delta.setZero(n_P); 
    
    for (int e=0; e<epochs; e++) {
        
        gE = getEnergyGradient(random,H,PL,HL);
        delta = moment*delta - lr*gE;
        updateParameters(PL,HL,delta);
        cout << "Epoch # " << e << "\tEnergy =  " << Energy;
        cout << "\tError = " << setprecision(8) << deltaEnergy;
        cout << endl;
    }
}



void NeuralQuantumState::optimizeNQS(MTRand & random,ofstream & file,
                   Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL) 
{

    






}

//void VariationalMC::update_SR(PsiLayer& PL, HiddenLayer& HL) {
//
//    VectorXd h;
//    double eL;
//    
//    h  = HL.forward_pass(spins);
//    eL = getLocalEnergy(PL,HL);
//    
//    int c = 0;
//        
//    //Updating O on W
//    for (int k=0; k<n_in; k++) {
//        for (int i=0; i<n_h; i++) {
//            O(c)  += h(i)*(1.0-h(i))*PL.Z(i)*spins(k);
//            Oe(c) += h(i)*(1.0-h(i))*PL.Z(i)*spins(k)*eL;
//            c++;
//        }
//    }
//    
//    //Updating O on b
//    for (int i=0; i<n_h; i++) {
//        O(c)  += h(i)*(1.0-h(i))*PL.Z(i);
//        Oe(c) += h(i)*(1.0-h(i))*PL.Z(i)*eL;
//        c++;
//    }
// 
//    // Updating O on Z
//    for (int i=0;i<n_h; i++) {
//        O(c)  += h(i);
//        Oe(c) += h(i)*eL;
//        c++;
//    }
//    
//    // Updating OO
//    for (int k=0; k<n_P; k++) {
//        for (int q=0; q<n_P; q++) {
//            OO(k,q) += O(k) * O(q);   
//        }
//    }
//   
//    //Updating the local Energy 
//    E += eL;
//
//}
//
//
//void VariationalMC::updateParameters_SR(PsiLayer& PL, HiddenLayer& HL, int e) {
//
//        
//    double mcs = 1.0*MCS;
//    int c = 0;
//    VectorXd delta;
//    VectorXd F;
//    MatrixXd S;
//    
//    F.setZero(n_P);
//    S.setZero(n_P,n_P);
//    delta.setZero(n_P); 
//    
//    double lambda0 = 100;
//    double b=0.9;
//    double lambdaMIN = 0.0001;
//    double lambda = lambda0*pow(b,e);
//
//    for (int k=0; k<n_P; k++) {
//        F(k) = 2*(Oe(k)/mcs - E*O(k)/(mcs*mcs));   
//    }
//    
//    for (int k=0; k<n_P; k++) {
//        for (int q=0; q<n_P; q++) {
//            S(k,q) = OO(k,q)/mcs - O(k)*O(q)/(mcs*mcs);
//        }   
//    }
//
//    for (int k=0; k<n_P; k++) {
//        S(k,k) += max(lambda,lambdaMIN)*S(k,k);
//    }
//
//    delta = S.jacobiSvd(ComputeThinU | ComputeThinV).solve(F);
//
//    for (int k=0; k<n_in; k++) {
//        for (int i=0; i<n_h; i++) {
//            HL.W(k,i) += -lr*delta(c); 
//            c++;
//        }
//    }
//    
//    for (int i=0; i<n_h; i++) {
//        HL.b(i) += -lr*delta(c);
//        c++;
//    }
//
//    for (int i=0; i<n_h; i++) {
//        PL.Z(i) += -lr*delta(c);
//        c++;
//    }        
//       
//}
//
//
//void VariationalMC::MC_run(MTRand& random, PsiLayer& PL, HiddenLayer& HL){
//    
//    int site;
//    double q;
//    double LogPsi;
//    double LogPsi_prime;
//    
//    for (int n=0; n<eq; n++) {
//        for (int k = 0; k< L; k++) {
//            site = random.randInt(L-1);
//            LogPsi = getLogPsi(PL,HL);
//            spins(site) *= -1;
//            LogPsi_prime = getLogPsi(PL,HL);
//            q = exp(2*(LogPsi_prime - LogPsi));
//            
//            if (random.rand() > q) {
//                spins(site) *= -1;
//            }
//        }
//    }
//
//    for (int n=0; n<MCS; n++) {
//        for (int k = 0; k< L; k++) {
//            site = random.randInt(L-1);
//            LogPsi = getLogPsi(PL,HL);
//            spins(site) *= -1;
//            LogPsi_prime = getLogPsi(PL,HL);
//            q = exp(2*(LogPsi_prime - LogPsi));
//            
//            if (random.rand() > q) {
//                spins(site) *= -1;
//            }
//        }
//        //update_SGD(PL,HL);
//        update_SR(PL,HL);
// 
//    }
//}
//
//void VariationalMC::train(MTRand & random, ofstream & file,
//                          PsiLayer& PL, HiddenLayer& HL) 
//{
//    
//    int lr_switch1 = int(epochs / 10);
//    int lr_switch2 = int(epochs / 2);
//    int record_frequency = 100;
//
//    for (int k=1; k<epochs+1; k++) {
//        reset(PL,HL);
//        MC_run(random,PL,HL);
//        
//        //updateParameters_SGD(PL,HL);
//        updateParameters_SR(PL,HL,k);
//        
//        Energy = E/(1.0*MCS*L);
//        
//        file << Energy << endl;
//        //cout << "Epoch: " << k << "   Ground State Energy: " << Energy << endl; 
//    }
//
//    //for (int k=1; k<lr_switch1+1; k++) {
//    //    reset(PL,HL);
//    //    MC_run(random,PL,HL);
//    //    updateParameters(PL,HL);
//    //    Energy = E/(1.0*MCS*L);
//    //    if ((k%record_frequency) == 0)
//    //        file << k << "      " << Energy << endl;
//    //    //cout << "Epoch: " << k << "   Ground State Energy: " << Energy << endl; 
//    //}
//    //
//    //lr /= 2.0;
//
//    //for (int k=lr_switch1; k<lr_switch2+1; k++) {
//    //    reset(PL,HL);
//    //    MC_run(random,PL,HL);
//    //    updateParameters(PL,HL);
//    //    Energy = E/(1.0*MCS*L);
//    //    if ((k%record_frequency) == 0)
//    //        file << k << "      " << Energy << endl;
//    //    //cout << "Epoch: " << k << "   Ground State Energy: " << Energy << endl; 
//    //}
//
//    //lr /= 5.0;
//
//    //for (int k=lr_switch2; k<epochs+1; k++) {
//    //    reset(PL,HL);
//    //    MC_run(random,PL,HL);
//    //    updateParameters(PL,HL);
//    //    Energy = E/(1.0*MCS*L);
//    //    if ((k%record_frequency) == 0)
//    //        file << k << "      " << Energy << endl;
//    //    //cout << "Epoch: " << k << "   Ground State Energy: " << Energy << endl; 
//    //}
//
//
//}
//
//////*****************************************************************************
////// Print Network Informations
//////*****************************************************************************
////
//void VariationalMC::printNetwork(PsiLayer& PL, HiddenLayer& HL) 
//{
//
//    cout << "\n\n******************************\n\n" << endl;
//    cout << "NEURAL NETWORK VARIATIONAL QUANTUM MONTE CARLO\n\n";
//    cout << "Machine Parameter\n\n";
//    cout << "\tNumber of Inputs Units: " << L << "\n";
//    cout << "\nNumber of Hidden Units: " << HL.n_h << "\n";
//    cout << "\tMagnetic Field: " << h << "\n";
//    cout << "\nHyper-parameters\n\n";
//    cout << "\tLearning Rate: " << lr << "\n";
//    cout << "\tEpochs: " << epochs << "\n";
//    cout << "\tMonte Carlo Steps: " << MCS << "\n";
//    cout << "\tInitial distribution width: " << HL.bound << "\n";
//    
//    
// 
//}
