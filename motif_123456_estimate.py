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
X = np.asarray(data['X'], dtype=int) - 1  # for easier index mapping
NUM_SAMPLE, LEN_MOTIF = X.shape
MAX_ITER = 200
# Sample from dirichlet distribution sum up to 1 and that's give us nice oneliner.
# thetaB = np.random.dirichlet(np.ones(NUM_OF_GENS), size=1)[0]
# theta = np.random.dirichlet(np.ones(NUM_OF_GENS), size=LEN_MOTIF).transpose()
thetaB = np.full(shape=NUM_OF_GENS, fill_value=1 / NUM_OF_GENS, dtype=np.float64)
theta = np.full(shape=(NUM_OF_GENS, LEN_MOTIF), fill_value=1 / NUM_OF_GENS, dtype=np.float64)

for _ in range(MAX_ITER):
    print(_)
    Q_0 = (1 - alpha) * np.prod(thetaB[X], axis=1)
    lambda_Q_0 = LEN_MOTIF * np.sum(Q_0)

    Q_1 = alpha * np.prod(theta[X, np.arange(LEN_MOTIF)], axis=1)
    lambda_Q_1 = np.sum(Q_1)

    for row_index in range(NUM_OF_GENS):
        X_ind = X == row_index
        thetaB[row_index] = np.sum(Q_0 * np.sum(X_ind, axis=1)) / lambda_Q_0
        for col_index in range(LEN_MOTIF):
            theta[row_index, col_index] = np.sum(Q_1[X_ind[:, col_index]]) / lambda_Q_1
    print((_, thetaB))

estimated_params = {
    "alpha": alpha,
    "Theta": theta.tolist(),
    "ThetaB": thetaB.tolist()
}

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)
