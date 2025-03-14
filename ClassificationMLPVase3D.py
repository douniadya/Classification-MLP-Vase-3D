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
#-----------------------------------------------------
    def sigmoid(self, x):     
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derive(self, x):
        return x * (1 - x)
 #---------------------------------------------------   
    def forward(self, X):
       A = X
       activation = [X]
       Z_values = []

       for poids, biais in zip(self.poids, self.biais):
           Z = np.dot(poids, A) + biais
           A = self.sigmoid(Z)
           Z_values.append(Z)
           activation.append(A)
        
         return A, activation, Z_values
 #----------------------------------------------------------   
    def backpropagation(self, X, y):
       activation, Z_values = self.forward(X)
       L= len(self.poids)
       delta =[None] * L
       #Calcul de l'erreur de la derniere couche
       delta[L-1] = (activation[L-1] - y) * self.sigmoid_derive(activation[L-1])
       #Calcul de l'erreur pour les autres couches
       for i in range(L-2, -1, -1):
           delta[i] = np.dot(self.poids[i+1].T, delta[i+1]) * self.sigmoid_derive(activation[i])
        
        #mise a jour des poids
       for i in range(L):
           self.poids[i] -= self.alpha * np.dot(delta[i], activation[i].T)
           self.biais[i] -= self.alpha * np.sum(delta[i], axis=1, keepdims=True)
 #-----------------------------------------------------------------       
    def train(self, X, y, epochs=1000):
       for ep in range(epochs):
           for X, y in zip(X, y):
               self.backpropagation(X, y)
            if ep % 100 == 0:
                print(f'Epoch {ep} finished')
    
    
