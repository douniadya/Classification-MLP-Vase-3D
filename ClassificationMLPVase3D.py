
import numpy as np
import matplotlib.pyplot as plt
import time

def printLine():
   print("\n--------------------------------------------------")

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


  
#--------------------------------------------------------------------
  def sigmoid(self, x):     
      return 1 / (1 + np.exp(-x))
#------------------------------------------------------------------------
  
  def sigmoid_derive(self, x):
      return x * (1 - x)
#------------------------------------------------------------------
   
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
    for i in range(iterations):
        self.retropropagation(X, y)
        #if i % 100==0:
         #   print("iteration", i, "/", iterations, ": done")
   
   
  def accuracy(self, p,y):
     pre = (p>=0.5).astype(int)
     yn = y.astype(int).flatten()
     eq = sum(pre==yn)
     acc= (eq /len(y))*100
     return round(float(acc), 2)

  def plot_vase_best(self, X,Y):
       # visualisation of the vase
       
   
       x = X[:, 0]
       y = X[:, 1]
       z = X[:, 2]

       # Create 3D scatter plot
       fig_b = plt.figure(figsize=(10, 8))
       ax_b = fig_b.add_subplot(111, projection='3d')
   
       Y= np.array(Y)       
       ax_b.scatter(x[Y==1],y[Y==1], z[Y==1], c='red', s=5)
    
       ax_b.set_xlabel('X')
       ax_b.set_ylabel('Y')
       ax_b.set_zlabel('Z')
       ax_b.set_title('3D Vase visualisation _ 98.8')
       plt.tight_layout()
       plt.show() 

#-------------------------------------- MAIN -----------------------------------------
 
def main():

   # Charger les données from file

   data_file = "data.txt"
   lines = 20000
   data = np.loadtxt(data_file)[:lines]   

   X_data = data[:, :3]   
   y_data = data[:, 3].reshape(-1, 1)   

   mlp =MLP(inputs=3,outputs=1, hidden_layers=[5,3], alpha=0.5)
    
   print("\n testing the ",lines," samples")
   """
   for i in range(lines):   
    output, _ = mlp.forward(X_data[i])
    print("Sample ",i,": DATA=", X_data[i],": True=", y_data[i]," output= ",output.flatten())
   """
    
   #mlp.train(X_data,y_data,20000)
   
   #print("output after retropropagation :")
   p=[]
   new_p=[]
   new_p= np.array(new_p)
   p, _ = mlp.forward(X_data)
   p = p.flatten()

   """
   for i in range(lines):   
      output, _ = mlp.forward(X_data[i])
      print("Sample ",i,": True=", y_data[i]," ","output= ",output.flatten())
      p.append(output.flatten()[0])
   """
   p = np.array(p)
   acc = mlp.accuracy(p,y_data)
   acc_max = acc
   """
   for i in range(7):
      print("iteration :",i+1)
      mlp.train(X_data,y_data,20000)
      p, _ = mlp.forward(X_data)
      p = p.flatten()
      acc_new = mlp.accuracy(p,y_data)

      if acc_new >acc :
         acc_max = acc_new
         acc=acc_new
         new_p = p
         printLine()
         print(f"MLP accuracy : {acc}")

   printLine()
   print(f"MLP accuracy (MAX): {acc_max}") 
   

   # visualisation of the vase
   
   x = X_data[:, 0]
   y = X_data[:, 1]
   z = X_data[:, 2]

   # Create 3D scatter plot
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   
   ax.scatter(x[new_p>=0.5],y[new_p>=0.5], z[new_p>=0.5], c='red', s=5)
   
   #ax.scatter(x[new_p<0.5], y[new_p<0.5], z[new_p<0.5], c='gray', s=5)
    
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('3D Vase visualisation')
   plt.tight_layout()
   plt.show() 
   
   predictions = (new_p >= 0.5).astype(int)

   # Combine X_data and predicted labels
   output_data = np.hstack((X_data, predictions.reshape(-1, 1)))

   # Save in the same format as data.txt: x y z predicted_label
   np.savetxt("test.txt", output_data, fmt="%.16f %.16f %.16f %d")
   """

   data_file_best = "test_tst.txt"
   lines = 20000
   data_b = np.loadtxt(data_file_best)[:lines]   

   X_data_b = data_b[:, :3]   
   y_data_b = data_b[:, 3]

   mlp.plot_vase_best(X_data_b,y_data_b)   

    
   #---- testing on XOR----------------------------------------------------------------------------------------------------------------------------
   """
   printLine()
   
   print("Learning on XOR:")
   mpl=MLP(inputs=2,outputs=1, hidden_layers=[5,3], alpha=0.6)
   XOR_X =np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])
   
   XOR_y=np.array([0,1,1,0]).reshape(-1,1)

   for i in range(4):   
    XORoutput, _ = mpl.forward(XOR_X[i])
    print("Sample ",i,": True=", XOR_y[i][0]," output= ",XORoutput.flatten()[0])

   
   mpl.train(XOR_X,XOR_y,2000)
       
   
   print("output after retropropagation :")

   for i in range(4):   
      outputXOR, _ = mpl.forward(XOR_X[i])
      print("Sample ",i,": True=", XOR_y[i][0]," ","output= ",outputXOR.flatten()[0])
   """

    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
