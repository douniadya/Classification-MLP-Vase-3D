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
    
    
