
import numpy as np

def coin_flip_model() -> None:
    # Indices of state space - number of 
    loss: int = 0
    draw: int = 1
    win: int = 2

    # Indices of observational space
    heads: int = 0
    tails: int = 1

    # Assumed probability of each outcome
    prior = np.array([0.25, 0.5, 0.25])
    print(f"Prior distro: {prior}")

    # Effect of each outcome on what coinflip is observed
    likelihood = np.array([
        [0.1, 0.9], # given loss 
        [0.5, 0.5], # given draw
        [0.7, 0.3], # given win
    ])
    print(f"Likelihood: {likelihood}")

    # Joint
    joint = likelihood * prior.reshape(3, 1)

    # Model evidence
    evidence = joint.sum(axis=0, keepdims=True)
    print(f"Evidence: {evidence}")

    # Posteriori
    post = joint / evidence
    print(f"Posteriori: {post}")
    
    # Surprise
    surprise = -np.log2(evidence)
    print(f"Surprise: {surprise}")

    # Bayesian surprise
    
if __name__ == "__main__":
    print("Entrypoint")
