import json

import numpy as np

import argparse
from motif_315688_generate import NUM_OF_GENS


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="generated_data.json", required=False,
                        help='Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default="estimated_params.json", required=False,
                        help='Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False,
                        help='Czy estymowac alpha czy nie?  (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha


input_file, output_file, estimate_alpha = ParseArguments()

with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)

alpha = data['alpha']
X = np.asarray(data['X'])
k, w = X.shape

# Sample from dirichlet distribution sum up to 1 and that's give us nice oneliner.
ThetaB = np.random.dirichlet(np.ones(NUM_OF_GENS), size=1)[0]
Theta = np.random.dirichlet(np.ones(NUM_OF_GENS), size=w).transpose()

estimated_params = {
    "alpha": alpha,
    "Theta": Theta.tolist(),
    "ThetaB": ThetaB.tolist()
}

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
