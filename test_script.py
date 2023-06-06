import numpy as np
import json
from motif_315688_estimate import X, EM, NUM_OF_GENS, LEN_MOTIF

if __name__ == "__main__":
    with open('params_set1.json', 'r') as inputfile:
        params = json.load(inputfile)
    theta_orig = np.asarray(params['Theta'])
    theta_b_orig = np.asarray(params['ThetaB'])


def d_tv(prob_orig, prob_estim):
    return 0.5 * np.sum(np.abs(prob_orig - prob_estim))


# random
print('random')
initial_theta = np.random.dirichlet(np.ones(NUM_OF_GENS), size=LEN_MOTIF).transpose()
initial_theta_b = np.random.dirichlet(np.ones(NUM_OF_GENS), size=1)[0]

estim_theta, estim_theta_b, loglik = EM(data=X, theta=initial_theta, thetaB=initial_theta_b)

total_distance = 1 / (LEN_MOTIF + 1) * (
        d_tv(prob_orig=theta_b_orig, prob_estim=estim_theta_b) + d_tv(prob_orig=theta_orig.flatten(),
                                                                      prob_estim=estim_theta.flatten()))

print(total_distance)

with open("random.csv", "a") as file:
    file.write(str(total_distance))
    file.write('\n')

print('uniform')

initial_thetaB = np.full(shape=NUM_OF_GENS, fill_value=1 / NUM_OF_GENS, dtype=np.float64)
initial_theta = np.full(shape=(NUM_OF_GENS, LEN_MOTIF), fill_value=1 / NUM_OF_GENS, dtype=np.float64)

estim_theta, estim_theta_b, loglik = EM(data=X, theta=initial_theta, thetaB=initial_theta_b)

total_distance = 1 / (LEN_MOTIF + 1) * (
        d_tv(prob_orig=theta_b_orig, prob_estim=estim_theta_b) + d_tv(prob_orig=theta_orig.flatten(),
                                                                      prob_estim=estim_theta.flatten()))
print(total_distance)

with open("uniform.csv", "a") as file:
    file.write(str(total_distance))
    file.write('\n')

print('bootstrap')

thetas = []
thetas_b = []
for _ in range(20):
    initial_theta = np.random.dirichlet(np.ones(NUM_OF_GENS), size=LEN_MOTIF).transpose()
    initial_theta_b = np.random.dirichlet(np.ones(NUM_OF_GENS), size=1)[0]

    estim_theta, estim_theta_b, loglik = EM(data=X, theta=initial_theta, thetaB=initial_theta_b)
    thetas.append(estim_theta)
    thetas_b.append(estim_theta_b)

stacked_thetas = np.stack(thetas, axis=0)
stacked_thetas_b = np.stack(thetas_b, axis=0)
estim_theta = np.mean(stacked_thetas, axis=0)
estim_theta_b = np.mean(stacked_thetas_b, axis=0)

total_distance = 1 / (LEN_MOTIF + 1) * (
        d_tv(prob_orig=theta_b_orig, prob_estim=estim_theta_b) + d_tv(prob_orig=theta_orig.flatten(),
                                                                      prob_estim=estim_theta.flatten()))
print(total_distance)

with open("bootstrap.csv", "a") as file:
    # Write new content to the file
    file.write(str(total_distance))
    file.write('\n')
