def main():
    # Create the probability distribution
    total_AiBi = 0
    prob = np.empty(A.shape[1])
    for i in range(A.shape[1]):
        prob[i] = np.linalg.norm(Algorithms.select(A[:, i])) * np.linalg.norm(Algorithms.select(B[i, :]))
        total_AiBi += prob[i]
    prob /= total_AiBi


if __name__ == '__main__':
    main()