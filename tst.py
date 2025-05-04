
import numpy as np
import pickle

class MLP:
    def __init__(self):
        self.W = []
        self.B = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        A = [X.T]
        for l in range(len(self.W)):
            Z = np.dot(self.W[l], A[-1]) + self.B[l]
            A.append(self.sigmoid(Z))
        return A[-1].T  # Return as (n_samples, 1)

    def load_weights(self, weight_file="weights.pkl", bias_file="biases.pkl"):
        with open(weight_file, 'rb') as f:
            self.W = pickle.load(f)
        with open(bias_file, 'rb') as f:
            self.B = pickle.load(f)
        print("Weights and biases loaded successfully.")

def apply_mlp_to_file(input_file="input.txt", output_file="output.txt"):
    # Load input X, Y, Z from file
    data = np.loadtxt(input_file)
    if data.shape[1] != 3:
        raise ValueError("Input file must contain exactly 3 columns (X Y Z)")

    X_data = data[:, :3]

    # Load trained model
    mlp = MLP()
    mlp.load_weights("weights.pkl", "biases.pkl")

    # Forward propagation
    predictions = mlp.forward(X_data)
    predictions = (predictions >= 0.5).astype(int).flatten()

    # Stack results with original data
    output_data = np.hstack((X_data, predictions.reshape(-1, 1)))

    # Save to file
    np.savetxt(output_file, output_data, fmt="%.6f %.6f %.6f %d")
    print(f"Predictions saved to: {output_file}")

if __name__ == "__main__":
    apply_mlp_to_file("test.txt", "test_predicted.txt")

#test.txt is the file that contains only the coordinates (x y z)
#test_predicted.txt is the output file