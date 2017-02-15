#include <stdlib.h>
#include <time.h>
#include <fstream>
#include "utilities.cpp"
#include "NQS.cpp"



int main(int argc, char* argv[]) {
   
     
    map<string,string> Helper;
    map<string,float> Parameters;
    map<string,string> Features;
    
    // Initialize the map of patameters 
    initializeParameters(Parameters);
    initializeFeatures(Features); 
    // Read the main command
    string command = argv[1];

    // In case of help, print the help commands
    if ((command.compare("-h") == 0) || (command.compare("-help") == 0)) {

        printHeader(); 
        exit(0);
    }
    
    // Read the model
    string model   = argv[2];
    
    // Read and store all the parameters
    get_parameter("D","Dimension",argc,argv,Parameters,Helper);
    get_parameter("L","System Size",argc,argv,Parameters,Helper);
    get_parameter("l","Layers",argc,argv,Parameters,Helper);
    get_parameter("nH","Hidden Units",argc,argv,Parameters,Helper);
    get_parameter("ep","Training steps",argc,argv,Parameters,Helper);
    get_parameter("lr","Learning Rate",argc,argv,Parameters,Helper);
    get_parameter("w","Width of the distribution for initial weights",argc,argv,Parameters,Helper);
    get_parameter("seed","Random Number Seed",argc,argv,Parameters,Helper);
    get_parameter("h","Magnetic Field",argc,argv,Parameters,Helper);
 
    // Read and store all the features
    get_feature("opt","Optimization Method",argc,argv,Features,Helper);
    get_feature("sH","Hamiltonian Sign",argc,argv,Features,Helper);
    get_feature("sNN","NN Sign",argc,argv,Features,Helper);
    get_feature("obj","Objective Function",argc,argv,Features,Helper);


    // Construct the random number generator class
    MTRand random(1234);
    
    // Construct the Output Layer class
    PsiLayer PL(random,Parameters["nH"],Parameters["w"]); 
 
    // Construct Hidden Layer class
    HiddenLayer HL(random,Parameters["L"],Parameters["nH"],Parameters["w"]);
    
    // Construct the Hamiltonian class
    Hamiltonian H(Parameters["D"],Parameters["L"],"chain"); 
    
    // Initialize the Hamiltonian
    //H.initialize(random,"Ferromagnetic",1.0,Features["sH"]); 
    H.initialize(random,Features["sH"]);

    // Construct the NQS class 
    NeuralQuantumState NQS(random,H.N,
                           Parameters["MCS"],
                           Parameters["lr"],
                           Parameters["ep"],
                           PL,HL,
                           Features["opt"],
                           Features["obj"],
                           Features["sNN"]); 
    
    ofstream fout("foo.txt");
    
    NQS.loadExactEnergy(model,int(Parameters["D"]));
    //NQS.loadExactWF(model,Features["sH"],int(Parameters["D"]));
    NQS.optimizeNQS(random,fout,H,PL,HL);
    
}
