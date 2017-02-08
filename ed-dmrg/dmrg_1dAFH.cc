#include "dmrg.h"
#include "sites/spinhalf.h"
#include "autompo.h"
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
using namespace itensor;
using namespace std;

int main(int argc, char *argv[]) {

    int N=4;
     
    //for (int i=1; i<argc; i++) {
    //    if (strcmp(argv[i],"--N") == 0) {
    //        N = atoi(argv[i+1]);
    //        cout << "N is " << N << endl;
    //    }
    //}

    // Initialize the sites making up the Hilbert space
    double energy;
    int MaxOrder = 50;
    auto sites = SpinHalf(N); //make a chain of N spin 1/2's
    auto ampo = AutoMPO(sites);

    // Set the parameters controlling the accuracy of the DMRG
    // calculation for each DMRG sweep. 
    // Here less than 5 cutoff values are provided, for example,
    // so all remaining sweeps will use the last one given (= 1E-10).
    auto sweeps = Sweeps(5);
    sweeps.maxm() = 10,20,100,100,200;
    sweeps.cutoff() = 1E-10;
    println(sweeps);

    ofstream file("energy_dmrg_1dAFH.dat");

    file << "L\t\tE\n";
    //file << 1 << "\t\t" << -0.5 << endl;

    for(int k=0; k<MaxOrder; k++) {
    	
	cout << "***********************************" << endl;
	cout << "Lattice size: " << N << endl;
	sites = SpinHalf(N);
	
	// Use the AutoMPO feature to create the 
    	// next-neighbor transverse field Ising model	
	for(int j = 1; j < N; ++j) {
    	    ampo += 1.0, "Sx",j,"Sx",j+1;
            ampo += 1.0 ,"Sy",j,"Sy",j+1;
	    ampo += 1.0, "Sz",j,"Sz",j+1;
    	}

        ampo += 1.0, "Sx",N,"Sx",1;
        ampo += 1.0, "Sy",N,"Sy",1;
	ampo += 1.0, "Sz",N,"Sz",1;

    	auto H = MPO(ampo);
		
    	// Initalize psi to be a random product
    	// MPS on the Hilbert space "sites"
    	auto psi = MPS(sites);

    	// Begin the DMRG calculation
    	energy = dmrg(psi,H,sweeps,"Quiet");

    	// Print the final energy reported by DMRG
    	printfln("\nGround State Energy = %.10f",energy/(1.0*N));
	//printfln("\nGround State Energy per Site = %.10f",energy/(1.0*N));

    	// Obtain the energy directly from the MPS
    	// and H by computing <psi|H|psi>
		//printfln("\nUsing psiHphi = %.10f", psiHphi(psi,H,psi) );

	file << N << "\t\t";
	file << setprecision(15) << energy/(1.0*N) << endl;

	ampo.reset();
        N += 2;	
    }
			
    return 0;
    
}
