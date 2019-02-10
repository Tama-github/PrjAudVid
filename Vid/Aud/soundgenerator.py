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
import sys
import time

class SoundGenerator :


    def __init__ (self, soundName) :
        fs, data = wavread(soundName)
        self.fs = fs
        self.data = self.scale(data, -1., 1.)
        self.totalTime = 0
        self.t = time.clock()
        self.frameWidth = 0

    def sound2ChanelsWlength (self, x, length, offset=0) :
        res = np.zeros(length)
        for i in range(length) :
            res[i] = x[(i+int(self.totalTime*self.fs)+offset)%len(x)]

        return res;

    def scale(self, X, x_min, x_max):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max() - X.min()
        
        return x_min + nom/denom 

    def testFiltrePH (self, x, seuil) :
        X = fftp.fft(x)
        
        X[:seuil] = 0.
        X[len(X)-seuil:] = 0.
        
        x = fftp.ifft(X).real
        return x


    def testFiltrePB (self, x, seuil) :
        X = fftp.fft(x)
        
        X[int(len(X)/2)-seuil : int(len(X)/2)+seuil] = 0.
        
        x = fftp.ifft(X).real
        return x

    def toneChanging (self, fs, x, coef) :
        t = len(x)/fs
        new_fs = int(fs*coef)
        new_x = sound2ChanelsWlength(x, int(t*new_fs))
        
        return new_fs, new_x

    def moduleChannelsAmplitude (self, left, right, coef) :
        #coef = coef;
        if coef < 0 :
            left = left + left*(-coef)
            right = right - right*(-coef)
        else :
            left = left - left*coef
            right = right + right*coef
        return left, right


    def getSoundFrom2Chan (self, left,right) :
        return np.array([left,right])

    def writeSound(self, fs, left, right, name) :
        wavwrite(name, fs, getSoundFrom2Chan(left,right));
    
    def spatialize (self, data, xPlace) :
        l = data
        r = l
        coef = (xPlace*2)/self.frameWidth - 1.
        l,r = self.moduleChannelsAmplitude(l,r,coef)
        return np.array([l,r])
    
    def genSampleFromObjects (self, objects, time) :
        tmp = np.zeros(int(time*self.fs))
        sample = self.getSoundFrom2Chan(tmp,tmp)
        for obj in objects :
            (box,center,vector) = obj
            (x,y) = center
            sig = self.spatialize(self.sound2ChanelsWlength(self.data,int(time*self.fs)), x)

            sample[0] = sample[0] + sig[0]
            sample[1] = sample[1] + sig[1]
        return sample
        
    def soundGenerationForFramePurpose (self, objects) :
        
        tmp = time.clock() - self.t
        self.totalTime = self.totalTime + tmp
        self.t = time.clock()
        
        sample = self.genSampleFromObjects (objects, tmp)
        
        if (len(sample) != 0) :
            self.scale(sample[0], -1., 1.)
            self.scale(sample[1], -1., 1.)
            
            save_stdout = sys.stdout
            sys.stdout = open('trash', 'w')            
            play(sample, self.fs)
            sys.stdout = save_stdout
        
        return sample
        
        
        

