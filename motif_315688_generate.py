import json 

import numpy as np
 
import argparse 

gen_dict = { 1: 'A', 2: 'C', 3: 'G', 4: 'T'}
NUM_OF_GENS = 4
 
# Musimy wczytać parametry

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output
    
    
param_file, output_file = ParseArguments()
 

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)
 

 
w=params['w']
k=params['k']
alpha=params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB =np.asarray(params['ThetaB'])


def generate_experiment_matrix(num_of_obserwation: int = k, length_of_motif : int = w):
    return np.random.uniform(size = (num_of_obserwation, length_of_motif))

def convert_prob_into_value(value, prog_vector): #da sie pewnie ładniej jakis refaktor poxniej
    gen = NUM_OF_GENS + 1
    for prog in reversed(prog_vector):
        if prog < value:
            break
        gen = gen - 1
    return gen


X = generate_experiment_matrix()

Z = np.random.binomial(n = 1, p = alpha, size = k)

for row_index, col_index in np.ndindex(X.shape):
    if Z[row_index] == 1:
        prob_vector = np.cumsum(Theta[:, col_index])
    else:
        prob_vector = np.cumsum(ThetaB)
    X[row_index, col_index] = convert_prob_into_value(X[row_index, col_index], prob_vector)

gen_data = {    
    "alpha" : alpha,
    "X" : X.tolist()
    }


with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)
 
