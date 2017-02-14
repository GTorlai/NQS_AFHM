#include "TFIM.h"

//*****************************************************************************
// Constructor 
//*****************************************************************************

Hamiltonian::Hamiltonian(int D_, int L_, string latticeType) 
{
    
    D = int(D_);
    L = int(L_);
    lattice = latticeType;
    N = pow(L,D);
    
    if (latticeType.compare("chain") == 0) {
        NN = 1; 
        NearestNeighbors.setZero(N);

        for (int i=0; i<N-1; i++) {
            NearestNeighbors(i) = i+1;
        }
        NearestNeighbors(N-1) = 0;
    }
    
    spins.setOnes(N); 
    
    //if (latticeType.compare("square") == 0) {
    //
    //}
    //
    //if (latticeType.compare("triangular") == 0) {
    //
    //}
    
    nL = NN * N;
}


//*****************************************************************************
// Initialize the Transverse-Field Ising Hamiltonian 
//*****************************************************************************

void Hamiltonian::initialize(MTRand & random,string interaction,double h_,string sign)
{

    if (interaction.compare("Ferromagnetic") == 0) {
        J = 1.0;
    }

    if (interaction.compare("Antiferromagnetic") ==0) {
        J = -1.0;
    }

    if (sign.compare("OFF") == 0) {
        h = h_;
    }

    if (sign.compare("ON") == 0) {
        h = -h_;
    }

    for (int i=0; i<N; i++) {
        if (random.rand() > 0.5)
            spins(i) = 1.0;
        else
            spins(i) = -1.0;
    }
}


//*****************************************************************************
// Print Information on the Lattice 
//*****************************************************************************

void Hamiltonian::printLatticeInfos()
{

    cout << "" << endl;

}


//*****************************************************************************
// Flip a spin at some site 
//*****************************************************************************

void Hamiltonian::flip(int& site) {

    spins(site) *= -1.0;

}


//*****************************************************************************
// Get the wavefunction 
//*****************************************************************************

Vector2d Hamiltonian::getWaveFunction(PsiLayer& PL, HiddenLayer& HL) 
{

    VectorXd h;
    Vector2d WF; 
    
    h = HL.forward_pass(spins);
    
    WF = PL.getWF(h);

    return WF;

}



//*****************************************************************************
// Compute the local energy 
//*****************************************************************************

double Hamiltonian::getLocalEnergy(PsiLayer& PL, HiddenLayer& HL) 

{

    double e = 0.0;
    Vector2d Psi;
    Vector2d Psi_flip; 
    
    // Compute Wavefunction
    Psi = getWaveFunction(PL,HL);

    for (int i=0; i<N; i++) {
        
        // Compute diagonal contribution at site i
        e += -J * spins(i)*spins(NearestNeighbors(i));
        
        // Flip spin at site i
        flip(i);

        // Compute the new wavefunction
        Psi_flip = getWaveFunction(PL,HL);
        
        // Compute off-diagonal contribution at site i
        e += -h * (Psi_flip(1) / Psi(1)) *exp(Psi_flip(0) - Psi(0));
        //e += -h * exp(Psi_flip(0) - Psi(0));
        // Flip the spin back
        flip(i);
    }
 
    return e;

}


//*****************************************************************************
// Performe on Monte-Carlo sweep for the TFIM 
//*****************************************************************************

void Hamiltonian::VQMC_sweep(MTRand & random, PsiLayer& PL, HiddenLayer & HL)
{

    Vector2d Psi;
    Vector2d Psi_prime;
    double q;

    int site;

    // Compute the wave-functions
    Psi = getWaveFunction(PL,HL);
 
    for (int i=0; i<N; i++) {
        
        // Pick a random site in the system
        site = random.randInt(N-1);
        flip(site);

        Psi_prime = getWaveFunction(PL,HL);

        q = ((Psi_prime(1)*Psi_prime(1))/(Psi(1)*Psi(1))) * exp(2*(Psi_prime(0) - Psi(0)));
        //q = exp(2*(Psi_prime(0) - Psi(0)));
        
        // REJECT
        if (random.rand() > q) {
            flip(site);
        }

        // ACCEPT
        else {
            Psi = Psi_prime;
        }
    }
}























//double NeuralQuantumState::getLocalEnergy(PsiLayer& PL, HiddenLayer& HL) {
//    
//    double e_ = 0.0;
//    Vector2d Psi_;
//    Vector2d Psi_flip_;
//    
//    Psi_ = getPsi(PL,HL);
//    
//    for (int i=0; i<L; i++) {
//        
//        e_ += spins(i)*spins(NN(i));
//        
//        if (spins(NN(i)) == (-1.0*spins(i))) {
//            spins(i)   *= -1.0;
//            spins(NN(i)) *= -1.0;
//            Psi_flip_ = getPsi(PL,HL);
//            
//            e_ += 0.5 * (Psi_(0)/Psi_flip_(0)) * (Psi_(1)/Psi_flip_(1));
//            
//            spins(i)   *= -1.0;
//            spins(NN(i)) *= -1.0;
//        }
//    }
//
//    //for (int i=0; i<L-1; i++) {
//    //    
//    //    e_ += spins(i)*spins(i+1);
//    //    
//    //    if (spins(i+1) == (-1.0*spins(i))) {
//    //        spins(i)   *= -1.0;
//    //        spins(i+1) *= -1.0;
//    //        Psi_flip_ = getPsi(PL,HL);
//    //        
//    //        e_ += 0.5 * (Psi_(0)/Psi_flip_(0)) * (Psi_(1)/Psi_flip_(1));
//    //        
//    //        spins(i)   *= -1.0;
//    //        spins(i+1) *= -1.0;
//    //    }
//    //}
//    //
//    //e_ += spins(L-1)*spins(0);
//    //    
//    //if (spins(0) == (-1.0*spins(L-1))) {
//    //    spins(L-1)   *= -1.0;
//    //    spins(0) *= -1.0;
//    //    Psi_flip_ = getPsi(PL,HL);
//    //    
//    //    e_ += 0.5 * (Psi_(0)/Psi_flip_(0)) * (Psi_(1)/Psi_flip_(1));
//    //    
//    //    spins(L-1)   *= -1.0;
//    //    spins(0) *= -1.0;
//    //}
// 
//    return e_;
//}
//

