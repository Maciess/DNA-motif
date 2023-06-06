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
    experiment_data = json.load(inputfile)

alpha = experiment_data['alpha']
X = np.asarray(experiment_data['X'], dtype=int) - 1  # for easier index mapping
NUM_SAMPLE, LEN_MOTIF = X.shape
MAX_ITER = 200
# Sample from dirichlet distribution sum up to 1 and that's give us nice oneliner.

initial_thetaB = np.full(shape=NUM_OF_GENS, fill_value=1 / NUM_OF_GENS, dtype=np.float64)
initial_theta = np.full(shape=(NUM_OF_GENS, LEN_MOTIF), fill_value=1 / NUM_OF_GENS, dtype=np.float64)


def EM(data=X, theta=initial_theta, thetaB=initial_thetaB):
    likelyhood_function = np.zeros(shape=MAX_ITER, dtype=np.float64)
    tol = 5e-5
    for _ in range(MAX_ITER):
        Q_0 = (1 - alpha) * np.prod(thetaB[data], axis=1)
        Q_1 = alpha * np.prod(theta[data, np.arange(LEN_MOTIF)], axis=1)

        common_denominator = Q_0 + Q_1
        Q_0 = Q_0 / common_denominator
        Q_1 = Q_1 / common_denominator

        likelyhood_function[_] = np.sum(Q_0 * np.log(Q_0) + Q_1 * np.log(Q_1))
        print(likelyhood_function[_])
        if _ > 0 and np.abs(likelyhood_function[_] - likelyhood_function[_ - 1]) < np.abs(
                tol * likelyhood_function[_ - 1]):
            print("COVERAGE")
            break

        lambda_Q_0 = LEN_MOTIF * np.sum(Q_0)
        lambda_Q_1 = np.sum(Q_1)

        for row_index in range(NUM_OF_GENS):
            X_ind = data == row_index
            thetaB[row_index] = np.sum(Q_0 * np.sum(X_ind, axis=1)) / lambda_Q_0
            for col_index in range(LEN_MOTIF):
                theta[row_index, col_index] = np.sum(Q_1[X_ind[:, col_index]]) / lambda_Q_1

    return theta, thetaB, likelyhood_function


if __name__ == "__main__":
    result_theta, result_theta_b, _ = EM()

    estimated_params = {
        "alpha": alpha,
        "Theta": result_theta.tolist(),
        "ThetaB": result_theta_b.tolist()
    }

    with open(output_file, 'w') as outfile:
        json.dump(estimated_params, outfile)
