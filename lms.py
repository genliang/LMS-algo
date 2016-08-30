#set alpha
#initialise w with random numbers
#set E to a large value (Emax)
#iter = 0 
#Repeat for until E < Emax or iter < maxIter
#  E = 0
#  for all training patterns {(x, d)}
     # output y = Wx
     # w = w + alpha*(d-y)*x

import numpy as np 
import matplotlib.pyplot as plt
import random


#weight_vector = np.zeros(3)
#w = np.array([0.1, 0.1, 0.1]) #25 weights

error_grp = []
iter_grp = []
#initialise the training patterns
training = [[np.array([-0.5, 1.2, -0.1]).transpose(), 0.2], [np.array([0.7, -0.5, -0.2]).transpose(), -0.8], [np.array([0.3, 1.2, 2.3]).transpose(), 0.8], 
            [np.array([1.2, 0.8, 1.0]).transpose(), 0.4], [np.array([-0.5, 1.2, -0.1]).transpose(), -0.2],
            [np.array([1.0, -0.3, 0.5]).transpose(), -0.1]]

def lms(w): 
  i = 0
  alpha = [0.1, 0.01, 0.001]
  Emax = 10000000000000000000000000
  maxIter = 500
  E = 0
  while ((i < maxIter) and (E < Emax)):
    E = 0
    for pair in training:
      y = np.dot(w.transpose(), pair[0])
      #print('y: ')
      #print(y)
      #print('==================')
      w = w + np.dot((alpha[2]*(pair[1] - y)), pair[0])
      #print('weight: ' + str(w))
      E = E + np.power((pair[1] - y), 2)
      #print('Error: ' + str(E))
    #Put the error in the array after going thru the whole training pattern
    error_grp.append(E)
    iter_grp.append(i)
    i = i + 1
  #print('i: ' + str(i))

  print('Final error: ' + str(E))
  print('final weight: ' + str(w))

w = []
for x in range(25):
  for y in range(3):
    val = round(random.random(), 1)
    print(val)
    w.append(val)
  weight_vector = np.array(w)
  print('Weight vector: ' + str(weight_vector))
  lms(weight_vector)
  w = []



  


#Draw the graph
plt.plot(iter_grp, error_grp)
plt.ylabel('Error')
plt.xlabel('No. of iterations')
plt.show()
#plt.savefig('0.01.png')