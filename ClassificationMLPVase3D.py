
import numpy as np

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
        w = np.random.randn(layers[i+1], layers[i]) * np.sqrt(2/(layers[i]+layers[i+1]))
        b = np.zeros((layers[i+1], 1))
        
        self.W.append(w)
        self.B.append(b)


  
#--------------------------------------------------------------------
  def sigmoid(self, x):     
      return 1 / (1 + np.exp(-x))
#------------------------------------------------------------------------
  
  def sigmoid_derive(self, x):
      return self.sigmoid(x) * (1 - self.sigmoid(x))
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

     err = (out - yt)* self.sigmoid_derive(out)
     for l in reversed(range(len(self.W))):
        gradW=np.dot(err,A[l].T)
        gradB =err
        self.W[l]=self.W[l] - self.alpha*gradW
        self.B[l]= self.B[l]- self.alpha* gradB

        if l>0 :
           err = np.dot(self.W[l].T, err)* self.sigmoid_derive(A[l])

          
#-------------------------------------- MAIN -----------------------------------------
 
def main():

   # Charger les données from file

   data_file = "data.txt"
   lines = 10
   data = np.loadtxt(data_file)[:lines]   

   X_data = data[:, :3]   
   y_data = data[:, 3].reshape(-1, 1)   

   mlp =MLP(inputs=3,outputs=1, hidden_layers=[4,3], alpha=0.01)
    
   print("\n testing the first ",lines," samples")

   for i in range(lines):   
    output, _ = mlp.forward(X_data[i])
    print("Sample ",i,": DATA=", X_data[i],": True=", y_data[i]," output= ",output.flatten())
   
   for e in range(10000):
     for i in range(lines):
       mlp.retropropagation(X_data[i],y_data[i])
   
   print("output after retropropagation :")

   for i in range(lines):   
      output, _ = mlp.forward(X_data[i])
      print("Sample ",i,": True=", y_data[i]," ","output= ",output.flatten())
    
   #---- testing on XOR----------------------------------------------------------------------------------------------------------------------------
   printLine()
   print("Learning on XOR:")
   mpl=MLP(inputs=2,outputs=1, hidden_layers=[4], alpha=0.1)
   XOR_X =np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])
   
   XOR_y=np.array([0,1,1,0]).reshape(-1,1)

   for i in range(4):   
    XORoutput, _ = mpl.forward(XOR_X[i])
    print("Sample ",i,": True=", XOR_y[i][0]," output= ",XORoutput.flatten()[0])

   for e in range(10000):
     for i in range(4):
       mpl.retropropagation(XOR_X[i],XOR_y[i])
   
   print("output after retropropagation :")

   for i in range(4):   
      output, _ = mpl.forward(XOR_X[i])
      print("Sample ",i,": True=", XOR_y[i][0]," ","output= ",output.flatten()[0])

   




    
if __name__ == "__main__":
    main()
