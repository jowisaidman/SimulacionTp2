import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import matrix_power

# A

# Estado 1 = suben ambas
# Estado 2 = baja A sube B
# Estado 3 = sube A baja B
# Estado 4 = bajan ambas

accionesA = pd.read_csv('accion A.csv')
accionesA.rename(columns={'Valor':'ValorA'}, inplace=True)

accionesB = pd.read_csv('accion B.csv')
accionesB.rename(columns={'Valor':'ValorB'}, inplace=True)

accionesA['DeltaA'] = accionesA.ValorA.shift(1)
accionesB['DeltaB'] = accionesB.ValorB.shift(1)
accionesA['DeltaA'] = accionesA.apply(lambda row : row['ValorA'] - row['DeltaA'], axis = 1)
accionesB['DeltaB'] = accionesB.apply(lambda row : row['ValorB'] - row['DeltaB'], axis = 1)

accionesA['PorcentajeA'] = accionesA.apply(lambda row : (row['DeltaA'] / row['ValorA']) * 100, axis = 1)
accionesB['PorcentajeB'] = accionesB.apply(lambda row : (row['DeltaB'] / row['ValorB']) * 100, axis = 1)

acciones = accionesA.merge(accionesB, how='inner',on='Dia')


def getEstado(deltaA, deltaB):
  if (deltaA >= 0 and deltaB >= 0):
    return 1
  if (deltaA < 0 and deltaB >= 0):
    return 2
  if (deltaA >= 0 and deltaB < 0):
    return 3
  if (deltaA < 0 and deltaB < 0):
    return 4

acciones["EstadoActual"] = acciones.apply(lambda row: getEstado(row['DeltaA'],row['DeltaB']), axis = 1)
acciones["EstadoAnterior"] = acciones.EstadoActual.shift(1)
# print(acciones.head(5))

def getMatrixTransitions(df):
  trans = np.zeros(shape=(4,4))
  transProb = np.zeros(shape=(4,4))
  for index,row in df.iterrows():
    if index > 2:
      trans[int(row[7]) - 1][int(row[8]) - 1] += 1
  for i in range(4):
    for j in range(4):
      transProb[i][j] = trans[i][j] / sum(list(trans[i]))
  return transProb

# B
transitionMatrix = getMatrixTransitions(acciones)
print(transitionMatrix)

# # C
[autovalores,autovectores] = LA.eig(transitionMatrix)
# como |lambak| < 1 con k-1 pi p = pi
print(autovectores)
print(np.transpose(autovectores))
print(autovalores)

# pelevado = autovectores * matrix_power(autovaloresMat,100) * matrix_power(np.transpose(autovectores),-1)
print(matrix_power(transitionMatrix,100))

#D

difAccionA = list(accionesA["DeltaA"])
difAccionA.pop(0)
plt.title("Diferencia Acciones A")
plt.hist(difAccionA,bins=100)
plt.show()

difAccionB = list(accionesB["DeltaB"])
difAccionB.pop(0)
plt.title("Diferencia Acciones B")
plt.hist(difAccionB,bins=100)
plt.show()

plt.figure(figsize=[12,4])

difAccionAPositivo = list(filter(lambda x : x >= 0,difAccionA));
plt.subplot(121)
plt.title("Diferencia positivas Acciones A")
plt.hist(difAccionAPositivo,bins='sturges')
plt.show()

plt.subplot(122)
difAccionAnegativa = list(filter(lambda x : x < 0,difAccionA));
plt.title("Diferencia negativas Acciones A")
plt.hist(difAccionAnegativa,bins='sturges');
plt.show()

plt.figure(figsize=[12,4])

difAccionBPositivo = list(filter(lambda x : x >= 0,difAccionB));
plt.subplot(121)
plt.title("Diferencia positivas Acciones B")
plt.hist(difAccionBPositivo,bins='sturges')
plt.show()

plt.subplot(122)
difAccionBnegativa = list(filter(lambda x : x < 0,difAccionB));
plt.title("Diferencia negativas Acciones B")
plt.hist(difAccionBnegativa,bins='sturges');
plt.show()

mediaSubaAccionA = np.mean(difAccionAPositivo)
mediaBajaAccionA = np.mean(difAccionAnegativa)
mediaSubaAccionB = np.mean(difAccionBPositivo)
mediaBajaAccionB = np.mean(difAccionBnegativa)

def getProxEstado(estadoAnterior,prob,matriz):
  probabilidadesDeTrnasicion=list(matriz[estadoAnterior])
  if prob <= probabilidadesDeTrnasicion[0]:
    return 0
  if prob <= sum(probabilidadesDeTrnasicion[0:1]):
    return 1
  if prob <= sum(probabilidadesDeTrnasicion[0:2]):
    return 2
  else:
    return 3


def getValorAccionA(valorAnterior,estadoProximo):
  if estadoProximo == 0 or estadoProximo == 2:
    delta = np.random.exponential(mediaSubaAccionA,1)[0]
  else:
    delta = -np.random.exponential(-mediaBajaAccionA, 1)[0]
  return valorAnterior+delta

def getValorAccionB(valorAnterior,estadoProximo):
  if estadoProximo == 0 or estadoProximo == 1:
    delta = np.random.exponential(mediaSubaAccionB,1)[0]
  else:
    delta = -np.random.exponential(-mediaBajaAccionB, 1)[0]
  return valorAnterior+delta


def simularCantDias(N,matriz):
  estadoActual = [0]
  valorAccionA = [list(accionesA['ValorA'])[-1]]
  valorAccionB = [list(accionesB['ValorB'])[-1]]
  for x in range(N):
    uniformeCambioEstado = np.random.uniform(0,1,1)[0]
    estadoProximo = getProxEstado(estadoActual[x],uniformeCambioEstado,matriz)
    valorAccionA.append(getValorAccionA(valorAccionA[-1],estadoProximo))
    valorAccionB.append(getValorAccionB(valorAccionB[-1],estadoProximo))
    estadoActual.append(estadoProximo)
  plt.figure(figsize=[12, 6])
  plt.title("Evolucion Acciones a lor largo de " + str(N) + " dias")
  plt.plot(np.arange(0,366,1),valorAccionA, label="Accion A")
  plt.plot(np.arange(0,366,1),valorAccionB, label="Accion B")
  plt.grid(True)
  plt.ylabel("+/- de la acción", fontsize=14)
  plt.xlabel("Días", fontsize=14)  # show a legend on the plot
  plt.legend()
  plt.show()

simularCantDias(365,transitionMatrix)

simularCantDias(365,transitionMatrix)