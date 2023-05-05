import json 

import numpy as np
 
import argparse 

 
# Musimy wczytać parametry

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False, help='Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False, help='Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False, help='Czy estymowac alpha czy nie?  (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha
    
    
input_file, output_file, estimate_alpha = ParseArguments()
 


with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)
 
 
 
alpha=data['alpha']
X= np.asarray(data['X'])
k,w = X.shape


# TO DO: GLOWNA CZESC: Wyestymuj Theta, ThetaB za pomoca EM i zapisz do output_file 
# Theta0 = wektor rozmiaru w
# Theta = macierz rozmiaru d na w = 4 na w
# przyklad losowy
ThetaB=np.zeros(4)
ThetaB[:(4-1)]=np.random.rand(4-1)/4
ThetaB[4-1]=1-np.sum(ThetaB)

Theta = np.zeros((4,w))
Theta[:(w),:]=np.random.random((3,w))/w
Theta[w,:]=1-np.sum(Theta,axis=0)


# ZADANIE BONUSOWE: jeśli estimate_alpha=="yes", to wtedy
# trzeba również estymować alpha (zignorować inormację otrzymaną z input_file)

estimated_params = {
    "alpha" : alpha,            # "przepisujemy" to alpha, one nie bylo estymowane 
    "Theta" : Theta.tolist(),   # westymowane
    "ThetaB" : ThetaB.tolist()  # westymowane
    }

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
    
    
    
