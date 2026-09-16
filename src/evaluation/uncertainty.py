import numpy as np


def entropy(probabilities):

    probabilities = np.asarray(
        probabilities
    )

    probabilities = np.clip(
        probabilities,
        1e-8,
        1.0,
    )

    return -np.sum(
        probabilities
        * np.log(probabilities),
        axis=-1,
    )


def confidence(probabilities):

    probabilities = np.asarray(
        probabilities
    )

    return np.max(
        probabilities,
        axis=-1,
    )