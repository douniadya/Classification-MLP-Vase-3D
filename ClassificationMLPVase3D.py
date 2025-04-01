# Classe pour la classification de donnees 3D TST

import numpy as np

class MLP:
  def __init__ (self, entree, sortie, couche_cache, alpha = 0.01):
    self.entree = entree
    self.sortie = sortie
    self.couche_cache = couche_cache
    self.alpha = alpha
    #initialisation des poids
    self.poids = []
    #initialisation des biais
    self.biais = []
    # Poids pour la couche d'entree vers la premiere couche cachee
    self.poids.append(np.random.randn(couche_cache[0], entree))
    self.biais.append(np.random.randn(couche_cache[0], 1))
    #poids entre couche cachee
    for i in range(1, len(couche_cache)):
        self.poids.append(np.random.randn(couche_cache[i], couche_cache[i-1]))
        self.biais.append(np.random.randn(couche_cache[i], 1))

    #poids entre la derniere couche cachee et la couche de sortie
    self.poids.append(np.random.randn(sortie, couche_cache[-1]))
    self.biais.append(np.random.randn(sortie, 1))

#--------------------------------------------------------------------
  def sigmoid(self, x):     
      return 1 / (1 + np.exp(-x))
#------------------------------------------------------------------------
  
  def sigmoid_derive(self, x):
      return x * (1 - x)
#------------------------------------------------------------------
  
  def forward(self, X):
         
        A = X.reshape(-1, 1)  # Ensure input is a column vector
        activation = [A]  # Store activations
        Z_values = []   

        for poids, biais in zip(self.poids, self.biais):
            Z = np.dot(poids, A) + biais
            A = self.sigmoid(Z)
            Z_values.append(Z)
            activation.append(A)

        return A, activation, Z_values   
#-------------------------------------------------------------------  
  def backpropagation(self, X, y):
         
        _, activation, Z_values = self.forward(X)  # Forward pass
        L = len(self.poids)  # Number of layers
        delta = [None] * L  # Store errors

        # Compute output layer error
        delta[L-1] = (activation[L] - y) * self.sigmoid_derive(activation[L])  

        # Compute errors for hidden layers
        for i in range(L-2, -1, -1):
            delta[i] = np.dot(self.poids[i+1].T, delta[i+1]) * self.sigmoid_derive(activation[i+1])  

        # Update weights and biases
        for i in range(L):
            self.poids[i] -= self.alpha * np.dot(delta[i], activation[i].T)  
            self.biais[i] -= self.alpha * np.sum(delta[i], axis=1, keepdims=True)   
#----------------------------------------------------------------------------------------------
  
  def train(self, X, y, epochs=1000):
    X = np.array(X)  #   Convert list to NumPy array
    y = np.array(y)   
    
    if X.ndim == 1:  # If X is a 1D array, reshape it
        X = X.reshape(-1, 1)

    if y.ndim == 1:   
        y = y.reshape(-1, 1)

    X = X.T  #  Transpose X to match weight dimensions
    y = y.T   

    for ep in range(epochs):
        for x_i, y_i in zip(X.T, y.T):  # Iterate over samples
            x_i = x_i.reshape(-1, 1)   
            y_i = y_i.reshape(-1, 1)   
            self.backpropagation(x_i, y_i)

        if ep % 100 == 0:
            print(f'Epoch {ep} finished')
          
#-------------------------------------- MAIN -----------------------------------------
 
def main():

   # Charger les données from file

   data_file = "data.txt"
   lines = 5
   data = np.loadtxt(data_file)[:lines]  # Read first 5 lines

   # Extract input features (first 3 columns) and labels (last column)
   X_data = data[:, :3]  # First three columns (x, y, z)
   y_data = data[:, 3].reshape(-1, 1)  # Last column (label), reshaped for MLP

   # Define an MLP for 3D input (3 input neurons, 1 hidden layer with 4 neurons, 1 output neuron)
   mlp = MLP(entree=3, sortie=1, couche_cache=[4], alpha=0.1)

   print(f" Testing on the first {lines} samples from data.txt")

   # Forward pass before training
   print("\n Outputs BEFORE training:")
   for i in range(lines):
       output, _, _ = mlp.forward(X_data[i])
       print(f"Sample {i}: X = {X_data[i]}, y = {y_data[i]}, Output = {output.flatten()}")

   # Training (one step per sample)
   for i in range(lines):
       mlp.backpropagation(X_data[i], y_data[i])

   # Forward pass after training
   print("\n Outputs AFTER training:")
   for i in range(lines):
       output, _, _ = mlp.forward(X_data[i])
       print(f"Sample {i}: X = {X_data[i]}, y = {y_data[i]}, Output = {output.flatten()}")

if __name__ == "__main__":
    main()
