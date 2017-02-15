#include "NQS.h"
#include <Eigen/Dense>
#include <algorithm>
#include <iomanip>
#include <bitset>

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
    
    //if (sign.compare("on") == 0) 
    n_P  = int(n_h*(n_in+3)+2);
    //else
    //    n_P  = int(n_h*(n_in+2));

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

void NeuralQuantumState::loadExactWF(string& model, string& sign,int D)
{

    string fileName;
    fileName = "data/ed-dmrg/wf_";
    fileName += boost::str(boost::format("%d") % D);
    fileName += "d";
    fileName += model;
    fileName += "_L";
    fileName += boost::str(boost::format("%d") % N);
    fileName += "_sign" + sign;
    fileName += ".txt";

    ifstream fin(fileName);

    double e;
    double temp; 
    
    int dim = pow(2,N);
    
    exactWF.setZero(dim);


    for (int i=0; i<dim; i++) {
        fin >> exactWF(i);
        
    }

}


double NeuralQuantumState::getOverlap(PsiLayer& PL, HiddenLayer& HL) 
{

    double overlap= 0.0;
    double temp=0.0;
    double Z = 0.0;
    int dim = pow(2,N);
    
    VectorXd spin;
    spin.setZero(N);

    Vector2d Psi;

    VectorXd psi;
    psi.setZero(dim);

    bitset<4> spinBit;

    for (int i=0; i<dim; i++){

        spinBit = i;
        
        for (int j=0; j<N; j++) {
            spin(N-1-j) = spinBit[j];
        }
        
        Psi = PL.getWF(HL.forward_pass(spin));
        
        psi(i) = exp(Psi(0));

        temp += psi(i) * exactWF(i);
        
        Z += psi(i)*psi(i);
    }

    overlap = temp/sqrt(Z);
    //overlap = psi(1) / sqrt(Z);
    return overlap;



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
    Vector2d Psi;
    double eL;
    double E=0.0;
    double sign=0.0; 
    int p;
    double eta = -0.25; 
    
    // Equilibration Stage
    for (int n=0; n<eq; n++) {
        H.VQMC_sweep(random,PL,HL); 
    } 
    
    // Compute statistics for the energy gradient
    for (int n=0; n<MCS; n++) {
        
        H.VQMC_sweep(random,PL,HL); 
        
        // Compute Hidden Layer state
        h = HL.forward_pass(H.spins);
        Psi = PL.getWF(h);
        
        // Compute Local Energy
        eL = H.getLocalEnergy(PL,HL);
        p = 0;
          
        // Accumulate statistics of O on W
        for (int k=0; k<n_in; k++) {
            for (int i=0; i<n_h; i++) {
                O(p)  += h(i)*(1.0-h(i))*PL.Z(i,0)*H.spins(k);
                //O(p)  += 2*h(i)*(1.0-h(i))*PL.Z(i,1)*H.spins(k)*(1.0-Psi(1));
                //O(p)  += -2.0*eta*(Psi(1)-1.0)*(Psi(1)-1.0)*Psi(1)*h(i)*(1.0-h(i))*PL.Z(i,1)*H.spins(k);
                Oe(p) += h(i)*(1.0-h(i))*PL.Z(i,0)*H.spins(k)*eL;
                //Oe(p) += 2*h(i)*(1.0-h(i))*PL.Z(i,1)*H.spins(k)*eL*(1.0-Psi(1));
                //Oe(p)  += -2.0*eta*(Psi(1)-1.0)*(Psi(1)-1.0)*Psi(1)*eL*h(i)*(1.0-h(i))*PL.Z(i,1)*H.spins(k); 
                p++;
            }
        }
        
        // Accumulate statistics of O on b
        for (int i=0; i<n_h; i++) {
            O(p)  += h(i)*(1.0-h(i))*PL.Z(i,0);
            //O(p)  += 2*h(i)*(1.0-h(i))*PL.Z(i,1)*(1.0-Psi(1));
            //O(p)  += -2*eta*h(i)*(1.0-h(i))*PL.Z(i,1)*(Psi(1)-1.0)*(Psi(1)-1.0)*Psi(1);
            Oe(p) += h(i)*(1.0-h(i))*PL.Z(i,0)*eL;
            //Oe(p) += 2*h(i)*(1.0-h(i))*PL.Z(i,1)*eL*(1.0-Psi(1));
            //Oe(p)  += -2*eta*h(i)*(1.0-h(i))*PL.Z(i,1)*(Psi(1)-1.0)*(Psi(1)-1.0)*Psi(1)*eL;
            p++;
        }
     
        // Accumulate statistics of O on Z
        for (int i=0;i<n_h; i++) {
            O(p)  += h(i);
            Oe(p) += h(i)*eL;
            p++;
        }
        //for (int i=0;i<n_h; i++) {
        //    O(p)  += 2*h(i)*(1.0-Psi(1));
        //    Oe(p) += 2*h(i)*eL*(1.0-Psi(1));
        //    //O(p)  += -2*eta*h(i)*(Psi(1)-1.0)*(Psi(1)-1.0)*Psi(1);
        //    //Oe(p) += -2*eta*h(i)*eL*(Psi(1)-1.0)*(Psi(1)-1.0)*Psi(1);
        //    
        //    p++;
        //}
        
        // Accumulate statistics of O on c
        O(p)  += 1;
        Oe(p) += eL;
        //p++;
        //O(p)  += 2*(1.0-Psi(1));
        //Oe(p) += 2*eL*(1.0-Psi(1));
 
 
        //cout << eL << endl << endl;
        // Accumulate statistics for energy    
        E += eL;
        sign += Psi(1);
    }
     
    Energy  = E/double(N*MCS);
    avgSign = sign/double(MCS);
    
    deltaEnergy = abs(Energy-ExactEnergy) / abs(ExactEnergy);

    // Compute the gradient of the energy
    for (int p=0; p<n_P; p++) {
        
        gE(p) = 2*(Oe(p)/(double(MCS))-O(p)*E/(double(MCS*MCS)));
    }

    return gE;
}


//*****************************************************************************
// Update all the NQS variational parameters 
//*****************************************************************************

void NeuralQuantumState::updateParameters(PsiLayer& PL, HiddenLayer& HL,
                                          VectorXd& DELTA) {
        
    int p = 0;

    for (int k=0; k<n_in; k++) {
        for (int i=0; i<n_h; i++) {
            HL.W(k,i) += DELTA(p); 
            p++;
        }
    }
    
    for (int i=0; i<n_h; i++) {
        HL.b(i) += DELTA(p);
        p++;
    }

    for (int i=0; i<n_h; i++) {
        PL.Z(i,0) += DELTA(p);
        p++;
    }
    
    for (int i=0; i<n_h; i++) {
        PL.Z(i,1) += DELTA(p);
        p++;
    }
    
    for (int j=0; j<2; j++) {
        PL.c(j) += DELTA(p);
        p++;
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
        cout << "Epoch # " << e << "\tEnergy =  " << Energy;
        cout << "\tError = " << setprecision(8) << deltaEnergy;
        cout << endl;
 
    }
}


void NeuralQuantumState::SGD_Momentum(MTRand & random, Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL)
{

    VectorXd gE(n_P);
    VectorXd delta;
    double overlap;
    delta.setZero(n_P); 
    
    for (int e=0; e<epochs; e++) {
        
        gE = getEnergyGradient(random,H,PL,HL);
        delta = moment*delta - lr*gE;
        updateParameters(PL,HL,delta);
        //overlap = getOverlap(PL,HL); 
        cout << "Epoch # " << e << "\tEnergy =  " << Energy;
        cout << "\tError = " << setprecision(8) << deltaEnergy;
        //cout << "\tAverage Sign = " << avgSign;
        //cout << "\tOverlap = " << overlap; 
        cout << endl;
    }
}


void NeuralQuantumState::SGD_AdaGrad(MTRand & random, Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL)
{

    VectorXd gE(n_P);
    ArrayXd g(n_P);
    VectorXd delta;
    ArrayXd deltaARRAY;
    double overlap;
    delta.setZero(n_P);
    deltaARRAY.setZero(n_P); 
    g.setZero(n_P);

    ArrayXd eps;
    eps.setOnes(n_P);
    eps = eps* 0.0000001;
    
    for (int e=0; e<epochs; e++) {
        
        gE = getEnergyGradient(random,H,PL,HL);
        
        g = g + gE.array() * gE.array();
        deltaARRAY = - lr*gE.array()/(g.sqrt()+eps);
        delta = deltaARRAY.matrix();
        updateParameters(PL,HL,delta);
        cout << "Epoch # " << e << "\tEnergy =  " << Energy;
        cout << "\tError = " << setprecision(8) << deltaEnergy;
        //cout << "\tAverage Sign = " << avgSign;
        //cout << "\tOverlap = " << overlap; 
        cout << endl;
    }
}


void NeuralQuantumState::SGD_AdaDelta(MTRand & random, Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL)
{

    VectorXd gE(n_P);
    ArrayXd g(n_P);
    ArrayXd x(n_P);
    VectorXd delta;
    ArrayXd deltaARRAY;
    double overlap;
    delta.setZero(n_P);
    deltaARRAY.setZero(n_P); 
    g.setZero(n_P);
    x.setZero(n_P);
    double gamma=0.99;
    ArrayXd eps;
    eps.setOnes(n_P);
    eps = eps* 0.0000001;
    
    for (int e=0; e<epochs; e++) {
        
        gE = getEnergyGradient(random,H,PL,HL);
        
        g = gamma*g + (1-gamma)* gE.array() * gE.array();
        deltaARRAY = -  ((x.sqrt()+eps)/(g.sqrt()+eps)) * gE.array();
        x = gamma*x + (1-gamma)* deltaARRAY * deltaARRAY;
        delta = deltaARRAY.matrix();
        updateParameters(PL,HL,delta);
        cout << "Epoch # " << e << "\tEnergy =  " << Energy;
        cout << "\tError = " << setprecision(8) << deltaEnergy;
        //cout << "\tAverage Sign = " << avgSign;
        //cout << "\tOverlap = " << overlap; 
        cout << endl;
    }
}

void NeuralQuantumState::optimizeNQS(MTRand & random,ofstream & file,
                   Hamiltonian& H,PsiLayer& PL, HiddenLayer& HL) 
{

    if (optimization.compare("SGD") == 0) {

        SGD_Plain(random,H,PL,HL);
    }

    if (optimization.compare("Momentum") == 0) {

        SGD_Momentum(random,H,PL,HL);
    }

    if (optimization.compare("AdaGrad") == 0) {

        SGD_AdaGrad(random,H,PL,HL);
    }

    if (optimization.compare("AdaDelta") == 0) {

        SGD_AdaDelta(random,H,PL,HL);
    }

    if (optimization.compare("SVRG") == 0) {

        //SVRG(random,H,PL,HL);
    }

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
