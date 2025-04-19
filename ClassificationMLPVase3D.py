
import numpy as np
import matplotlib.pyplot as plt

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
  
  def forward(self, x):
     a = x.reshape(-1, 1)
     A = [a]

     for l in range(len(self.W)):
        a = self.forwardL(a,l)
        A.append(a)

     return a, A
  
  def forwardL(self, inA, l):
     w = self.W[l]
     b = self.B[l]

     a = self.sigmoid(np.dot(w, inA)+b)

     return a
  
  #-----------------------------------------

  def retropropagation(self, x, yt):
     out, A = self.forward(x) 
     yt = yt.reshape(-1,1)
 
     D = [(out - yt) * self.sigmoid_derive(out)]
     for l in reversed(range(len(self.W)-1)):
        err = np.dot(self.W[l+1].T, D[-1])
        D.append(err * self.sigmoid_derive(A[l+1]))
     D.reverse()

     for l in range(len(self.W)):
        gradW=np.dot(D[l],A[l].T)
        gradB =D[l]
        self.W[l] -= self.alpha * gradW
        self.B[l] -= self.alpha * gradB
   
  def train(self, X,y,iterations):
     for i in range(iterations):
       for j in range(len(X)):
          self.retropropagation(X[j],y[j])
       if len(X) >10:
          print("iteration ",i,"/",iterations,": done")
 
    

#-------------------------------------- MAIN -----------------------------------------
 
def main():

   # Charger les données from file

   data_file = "data.txt"
   lines = 500
   data = np.loadtxt(data_file)[:lines]   

   X_data = data[:, :3]   
   y_data = data[:, 3].reshape(-1, 1)   

   mlp =MLP(inputs=3,outputs=1, hidden_layers=[10,7,5,3], alpha=0.01)
    
   print("\n testing the first ",lines," samples")

   for i in range(lines):   
    output, _ = mlp.forward(X_data[i])
    print("Sample ",i,": DATA=", X_data[i],": True=", y_data[i]," output= ",output.flatten())
   
    
   mlp.train(X_data,y_data,2000)
   
   print("output after retropropagation :")
   p=[]

   for i in range(lines):   
      output, _ = mlp.forward(X_data[i])
      print("Sample ",i,": True=", y_data[i]," ","output= ",output.flatten())
      p.append(output.flatten()[0])
   
    
   #---- testing on XOR----------------------------------------------------------------------------------------------------------------------------
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

   # visualisation of the vase
   
   x = X_data[:, 0]
   y = X_data[:, 1]
   z = X_data[:, 2]

   # Create 3D scatter plot
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   for i in range(len(x)):
       if p[i] >= 0.5:
           ax.scatter(x[i],y[i], z[i], c='red', s=5)
       else:
           ax.scatter(x[i], y[i], z[i], c='gray', s=5)
    
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('3D Vase visualisation')
   plt.tight_layout()
   plt.show() 
   



    
if __name__ == "__main__":
    main()
