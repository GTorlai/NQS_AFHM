#include "AFHM.h"

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

void Hamiltonian::initialize(MTRand & random,string sign)
{

    if (sign.compare("OFF") == 0) {
        Jxy = -1.0;
    }

    if (sign.compare("ON") == 0) {
        Jxy = 1.0;
    }

    for (int i=0; i<N/2; i++) {
        //if (random.rand() > 0.5)
        //    spins(i) = 1.0;
        //else
        spins(2*i)   = 1.0;
        spins(2*i+1) = -1.0;
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

void Hamiltonian::flip(int& site1,int& site2) {

    spins(site1) *= -1.0;
    spins(site2) *= -1.0;

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
        e += 0.25*spins(i)*spins(NearestNeighbors(i));
        
        if (spins(NearestNeighbors(i)) * spins(i) < 0) {
            
            // Flip spin at site i
            flip(i,NearestNeighbors(i));

            // Compute the new wavefunction
            Psi_flip = getWaveFunction(PL,HL);
        
            // Compute off-diagonal contribution at site i
            e += 0.5 * Jxy * (Psi_flip(1) / Psi(1)) *exp(Psi_flip(0) - Psi(0));
            
            // Flip the spin back
            flip(i,NearestNeighbors(i));
        }
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
    int index;

    int site1,site2;

    // Compute the wave-functions
    Psi = getWaveFunction(PL,HL);
 
    for (int i=0; i<N; i++) {
        
        do {
            // Pick two random site in the system
            site1 = random.randInt(N-1);
            site2 = random.randInt(N-1);
            index = spins(site1) * spins(site2);
        } while (index > 0);

        flip(site1,site2);

        Psi_prime = getWaveFunction(PL,HL);

        q = ((Psi_prime(1)*Psi_prime(1))/(Psi(1)*Psi(1))) * exp(2*(Psi_prime(0) - Psi(0)));
        //q = exp(2*(Psi_prime(0) - Psi(0)));
        
        // REJECT
        if (random.rand() > q) {
            flip(site1,site2);
        }

        // ACCEPT
        else {
            Psi = Psi_prime;
        }
    }
}



