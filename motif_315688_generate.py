import json
import numpy as np
import argparse

NUM_OF_GENS = 4


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False,
                        help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False,
                        help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output


param_file, output_file = ParseArguments()

if __name__ == "__main__":
    with open(param_file, 'r') as inputfile:
        params = json.load(inputfile)

    w = params['w']
    k = params['k']
    alpha = params['alpha']
    Theta = np.asarray(params['Theta'])
    ThetaB = np.asarray(params['ThetaB'])


    def generate_experiment_matrix(num_of_observation: int = k, length_of_motif: int = w):
        return np.empty(shape=(num_of_observation, length_of_motif), dtype=np.float64)


    X = generate_experiment_matrix()

    Z = np.random.binomial(n=1, p=alpha, size=k)

    for row_index, col_index in np.ndindex(X.shape):
        if Z[row_index] == 1:
            prob_vector = Theta[:, col_index]
        else:
            prob_vector = ThetaB
        X[row_index, col_index] = np.random.choice(NUM_OF_GENS, size=1, p=prob_vector) + 1

    gen_data = {
        "alpha": alpha,
        "X": X.tolist()
    }

    with open(output_file, 'w') as outfile:
        json.dump(gen_data, outfile)
