#ifndef UTILITIES_H
#define UTITLITES_H

#include <Eigen/Core>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <map>
#include <boost/format.hpp>

using namespace std;

//*****************************************************************************
// Get command line options
//*****************************************************************************

void initializeParameters(map<string,float>& par) 
{
    par["D"] = 1;
    par["L"] = 4;
    par["nH"] = 4;
    par["nH2"] = 2;
    par["nH3"] = 2;
    par["lr"] = 0.01;
    par["w"] = 1.0;
    par["l"] = 1;
    par["ep"] = 10;
    par["MCS"] = 1000; 
    par["seed"] = 1234;
}

void initializeFeatures(map<string,string>& feat) 
{
    feat["opt"]  = "SGD";
    feat["obj"]  = "Energy";
    feat["sH"] = "OFF";
    feat["sNN"] = "OFF";
}

 
//*****************************************************************************
// Read and store the parameters from the command line
//*****************************************************************************

void get_parameter(const string& arg, const string& description,
                int argc, char** argv, 
                map<string,float>& par, map<string,string>& helper)
{
    string flag = "-" + arg ;
    for (int i=3; i<argc; ++i) {
            
        if (flag.compare(argv[i]) ==0) {

            par[arg] = atof(argv[i+1]);
 
            break;
        }
    }

    helper[arg] = description;
}

//*****************************************************************************
// Read and store the features from the command line
//*****************************************************************************

void get_feature(const string& arg, const string& description,
                int argc, char** argv, 
                map<string,string>& feat, map<string,string>& helper)
{
    string flag = "-" + arg ;
    for (int i=3; i<argc; ++i) {
            
        if (flag.compare(argv[i]) ==0) {

            feat[arg] = argv[i+1];
 
            break;
        }
    }

    helper[arg] = description;
}



//*****************************************************************************
// Generate the Base Name of the simulation
//*****************************************************************************

string buildBaseName(const string& network, const string& model,
                     map<string,float>& par)
{

    string baseName = "NQS_";
    
    baseName += boost::str(boost::format("%.d") % int(par["l"]));
    baseName += "layer_nH";

    for (int l=1; l<=int(par["l"]); l++) {
        string ind = "nH";
        ind += boost::str(boost::format("%.d") % l);
        baseName += boost::str(boost::format("%.d") % int(par[ind]));
    }
            
    baseName += "_lr";
    baseName += boost::str(boost::format("%.3f") % par["lr"]);
    baseName += "_S";
    baseName += boost::str(boost::format("%d") % int(par["s"]));
 
    baseName += "_";
    baseName += model;
    baseName += "_L";
    //int L = int(sqrt(par["nV"]));
    //int L = int(par["nV"]);
    baseName += boost::str(boost::format("%d") % int(par["L"]));
    
    return baseName;
}

//*****************************************************************************
// Generate the Name of the model file
//*****************************************************************************

string buildModelName(const string& network, const string& model,
                     map<string,float>& par)
{
    
    //int L = int(sqrt(par["nV"]));
    //int L = par["nV"];
    
    string modelName = "data/networks/";
    modelName += model + "/";
    modelName += "L";
    modelName += boost::str(boost::format("%d") % int(par["L"]));
    modelName += "/";
    modelName += buildBaseName(network,model,par); 
    //modelName += "_B";
    //modelName += boost::str(boost::format("%.2f") % par["B"]);
    modelName += "_model.txt";
 
    return modelName;
}



string buildObserverName(const string& network, const string& model,
                     map<string,float>& par)
{
    
    string modelName = "data/training/";
    modelName += model + "/";
    modelName += buildBaseName(network,model,par); 
    modelName += "_B";
    modelName += boost::str(boost::format("%.2f") % par["B"]);
    modelName += "_gradient.txt"; 
    return modelName;
}

//*****************************************************************************
// Generate the Name of the model file
//*****************************************************************************

string buildMeasurementsName(const string& network, const string& model,
                     map<string,float>& par)
{
    
    string modelName = "data/measurements/";
    modelName += model + "/RBM/";
    modelName += buildBaseName(network,model,par); 
    //modelName += "_B";
    //modelName += boost::str(boost::format("%.2f") % par["B"]);
 
    return modelName;
}


//*****************************************************************************
// Print the executable header
//*****************************************************************************

void printHeader() 
{

    cout << "\n***************************************************\n";
    cout << "\nNeural Quantum State simulation of many-body systems\n\n";

}



//*****************************************************************************
// Print Matrix or Vector on the screen
//*****************************************************************************

template<typename T> 
ostream& operator<< (ostream& out, const Eigen::MatrixBase<T>& M)
{    
    for (size_t i =0; i< M.rows(); ++i) {
        
        for (size_t j =0; j< M.cols(); ++j) {
            
            out << M(i,j)<< " ";
        }
        
        if (M.cols() > 1) out << endl;
    }
    
    out << endl;

    return out;
}


//*****************************************************************************
// Write Matrix or Vector on file 
//*****************************************************************************

template<typename T> 
void write (ofstream& fout,const Eigen::MatrixBase<T>& M)
{
    for (size_t i =0; i< M.rows(); ++i) {
        
        for (size_t j =0; j< M.cols(); ++j) {
            
            fout << M(i,j)<< " ";
        }
        
        if (M.cols() > 1) fout << endl;
    }
    
    fout << endl;
}



#endif
