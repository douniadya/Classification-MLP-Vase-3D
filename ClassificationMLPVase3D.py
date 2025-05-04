
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

def printLine():
   print("\n--------------------------------------------------")
"""
@njit
def sigmoid(x):     
      return 1 / (1 + np.exp(-x))
#------------------------------------------------------------------------
@njit  
def sigmoid_derive(x):
      return x * (1 - x)
#------------------------------------------------------------------
"""  
class MLP:
  def __init__(self, inputs, outputs, hidden_layers, alpha=0.01):
     self.inputs = inputs
     self.outputs = outputs
     self.hidden_layers = hidden_layers
     self.alpha = alpha
     self.W = []
     self.B = []

     layers = [inputs] + hidden_layers + [outputs]

     for i in range(len(layers)-1):
        w = np.random.randn(layers[i+1], layers[i]) * np.sqrt(2./layers[i]) 
        b = np.zeros((layers[i+1], 1))
        
        self.W.append(w)
        self.B.append(b)

  def sigmoid(self,x):     
      return 1 / (1 + np.exp(-x))
#------------------------------------------------------------------------
  
  def sigmoid_derive(self,x):
      return x * (1 - x)
  
#--------------------------------------------------------------------
 
  def forward(self, X):
    A = [X.T]  
    for l in range(len(self.W)):
        Z = np.dot(self.W[l], A[-1]) + self.B[l]
        A.append(self.sigmoid(Z))
    return A[-1], A
  
  #-----------------------------------------

  def retropropagation(self, X, Y):
    Y = Y.T   
    out, A = self.forward(X)

    D = [(out - Y) * self.sigmoid_derive(out)]
    for l in reversed(range(len(self.W) - 1)):
        err = np.dot(self.W[l + 1].T, D[-1])
        D.append(err * self.sigmoid_derive(A[l + 1]))
    D.reverse()

    nem = X.shape[0]
    for l in range(len(self.W)):
        gradW = np.dot(D[l], A[l].T) / nem
        gradB = np.sum(D[l], axis=1, keepdims=True) / nem
        self.W[l] -= self.alpha * gradW
        self.B[l] -= self.alpha * gradB
   
  def train(self, X, y, iterations):
    for j in range(iterations):
        batch_size = 256
        for i in range(0, X.shape[0], batch_size):
          X_batch = X[i:i+batch_size]
          y_batch = y[i:i+batch_size]
          self.retropropagation(X_batch, y_batch)
        if j % 1000==0:
            print("iteration", j, "/", iterations, ": done")
   
   
  def accuracy(self, p,y):
     pre = (p>=0.5).astype(int)
     yn = y.astype(int).flatten()
     eq = sum(pre==yn)
     acc= (eq /len(y))*100
     return round(float(acc), 2)
  
  def save_weights(self, weight_file="weights.pkl", bias_file="biases.pkl"):
      with open(weight_file, 'wb') as f:
          pickle.dump(self.W, f)
      with open(bias_file, 'wb') as f:
          pickle.dump(self.B, f)
      print("Weights and biases saved successfully.")

  def load_weights(self, weight_file="weights.pkl", bias_file="biases.pkl"):
      with open(weight_file, 'rb') as f:
          self.W = pickle.load(f)
      with open(bias_file, 'rb') as f:
          self.B = pickle.load(f)
      print("Weights and biases loaded successfully.")
 

#-------------------------------------- MAIN -----------------------------------------
 
def main():

   data_file = "data.txt"
   lines = 20000
   data = np.loadtxt(data_file)[:lines]   

   X_data = data[:, :3]   
   y_data = data[:, 3].reshape(-1, 1)   

   mlp =MLP(inputs=3,outputs=1, hidden_layers=[10,36,12], alpha=0.5)
   print("\nTesting the ",lines," samples")
   
   p=[]
   new_p=[]
   new_p= np.array(new_p)
   p, _ = mlp.forward(X_data)
   p = p.flatten()

   acc = mlp.accuracy(p,y_data)
   acc_max = acc
   W = []
   B = []

   for i in range(2):
      print("Iteration :",i+1)
      mlp.train(X_data,y_data,8000)
      p, _ = mlp.forward(X_data)
      p = p.flatten()
      acc_new = mlp.accuracy(p,y_data)

      if acc_new > acc:
         acc_max = acc_new
         acc = acc_new
         new_p = p
         W = mlp.W.copy()
         B = mlp.B.copy()
         printLine()
         print(f"MLP accuracy : {acc}")

   printLine()
   print(f"MLP accuracy (MAX): {acc_max}") 

   # Save the best weights and biases
   mlp.W = W
   mlp.B = B
   mlp.save_weights("weights.pkl", "biases.pkl")

   # Visualize the vase
   x = X_data[:, 0]
   y = X_data[:, 1]
   z = X_data[:, 2]
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(x[new_p>=0.5], y[new_p>=0.5], z[new_p>=0.5], c='red', s=5)
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('3D Vase visualisation')
   plt.tight_layout()
   plt.show() 
   """   
   # ------------ TESTING ON test.txt --------------------
   printLine()
   print("Testing on test.txt (without labels)")

   test_data_file = "test.txt"
   output_file = "test_predicted.txt" # output file
   test_data = np.loadtxt(test_data_file)  # test.txt has only x,y,z

   predictions, _ = mlp.forward(test_data)
   predictions = (predictions >= 0.5).astype(int)  # binary output: 0 or 1

   # Save the results with predicted labels
   output_data = np.hstack((test_data, predictions.reshape(-1, 1)))
   np.savetxt(output_file, output_data, fmt="%.5f", delimiter=" ")

   print(f"Predicted labels saved to '{output_file}'")
   """

   
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
