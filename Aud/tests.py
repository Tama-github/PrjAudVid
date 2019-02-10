import numpy as np
from scipy.io.wavfile import write as wavwrite
from scipy.io.wavfile import read as wavread
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import scipy.fftpack as fftp
import pylab as py
import math
from audiolab import play
import time


def sound2ChanelsWlength (x, length, offset=0) :
    res = np.zeros(length)
    for i in range(length) :
      res[i] = x[(i+offset)%len(x)]

    return res;

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max() - X.min()
    
    return x_min + nom/denom 
  
def testFiltrePH (x, seuil) :
  X = fftp.fft(x)
  
  plt.figure()
  plt.plot(X)
  
  X[:seuil] = 0.
  X[len(X)-seuil:] = 0.
  
  plt.figure()
  plt.plot(X)
  
  x = fftp.ifft(X).real
  return x

def testFiltrePB (x, seuil) :
  X = fftp.fft(x)
  
  plt.figure()
  plt.plot(X)
  
  print(int(len(X)/2)-seuil)
  print(int(len(X)/2)+seuil)
  
  X[int(len(X)/2)-seuil : int(len(X)/2)+seuil] = 0.
  
  plt.figure()
  plt.plot(X)
  
  x = fftp.ifft(X).real
  return x
  
def toneChanging (fs, x, coef) :
  t = len(x)/fs
  new_fs = int(fs*coef)
  new_x = sound2ChanelsWlength(x, int(t*new_fs))
  
  plt.figure()
  plt.plot(new_x)
  
  return new_fs, new_x

def moduleChannelsAmplitude (left, right, coef) :
  #coef = coef;
  if coef < 0 :
    left = left + left*(-coef)
    right = right - right*(-coef)
  else :
    left = left - left*coef
    right = right + right*coef
  return left, right

def getSoundFrom2Chan (left,right) :
    return np.array([left,right]).transpose()
    
def writeSound(fs, left, right, name) :
  wavwrite(name, fs, getSoundFrom2Chan(left,right));
  

fs, data = wavread('lb_idle.wav')

#plt.figure()
#plt.plot(data)

#left = sound2ChanelsWlength (data, 100000)

#plt.figure()
#plt.plot(left)

#leftF = testFiltrePB(left, 46000)

#plt.figure()
#plt.plot(leftF)

#leftF = scale(leftF, -1, 1)

#plt.figure()
#plt.plot(leftF)

#nfs, nx = toneChanging (fs, leftF, 1.5)
#wavwrite('test.wav', nfs, nx)

#l = leftF
#r = l


#l, r = moduleChannelsAmplitude(l, r, -0.5)

#plt.figure()
#plt.plot(l)
  
#plt.figure()
#plt.plot(r)

#res = np.array([l,r]).transpose()

#wavwrite('test.wav', nfs, res)
#writeSound(fs, l, r, 'test.wav')

py.figure()
py.plot(data)


data = scale(data, -1., 1.)


totalTime = 0;
t = time.clock()
while (1) :
    t = time.clock() - t
    samples = sound2ChanelsWlength(data,int(t*fs),int(totalTime*fs))
    print (totalTime)
    print (t);
    print (samples)
    
    totalTime = totalTime + t
    if (len(samples) != 0) :
        scale(samples, -1., 1.)
        play(getSoundFrom2Chan(data,data).transpose(), fs)
    


