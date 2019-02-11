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
        self.data = self.scale(data, -0.5, 0.5)
        self.totalTime = 0
        self.t = time.clock()
        self.frameWidth = 0
        self.lastFrame = np.array([[0.,0.],[0.,0.]])

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
        
        tmp1 = X[seuil:int(len(X)/2.)]
        tmp2 = X[int(len(X)/2.):len(X)-seuil]
        
        Y = X*0.
        Y[:int(len(Y)/2.)-seuil] = tmp1
        Y[int(len(Y)/2.)+seuil:] = tmp2
        
        y = fftp.ifft(Y).real
        return y


    def testFiltrePB (self, x, seuil) :
        X = fftp.fft(x)
        
        tmp1 = X[:int(len(X)/2)-seuil]
        tmp2 = X[int(len(X)/2)+seuil:]
        
        Y = X*0.
        
        Y[seuil:int(len(Y)/2.)] = tmp1
        Y[int(len(Y)/2.):len(Y)-seuil] = tmp2
        
        y = fftp.ifft(Y).real
        return y

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

    def amplitudeChanging (self, sound2Chan, coef) :
        return sound2Chan * coef


    def getSoundFrom2Chan (self, left, right) :
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
        print sample
        return sample
        
    def smoothenStart (self, sample, nb) :
        
        for i in range(nb) :
            sample[0][i] = (sample[0][i]+self.lastFrame[0][len(self.lastFrame)-tmp+i])/2.
            sample[1][i] = (sample[1][i]+self.lastFrame[1][len(self.lastFrame)-tmp+i])/2.
        return sample
        
    
    def soundGenerationForFramePurpose (self, objects) :
        
        tmp = time.clock() - self.t
        sample = []
        if tmp >= 1./12 : # for each frame we want the previous sound to be finished before lauching a new one
            self.totalTime = self.totalTime + tmp
            self.t = time.clock()
            
            sample = self.genSampleFromObjects (objects, tmp)
            
            #if (self.lastFrame[0][0] == sample[0][0]) :
            #    print ("same as last frame")
            #else :
            #    print ("diferent sample")
            
            
            
            if (len(sample) != 0) :
                
                self.smoothenStart(sample,int(len(sample)/50.))
                self.scale(sample[0], -1., 1.)
                self.scale(sample[1], -1., 1.)
                
                save_stdout = sys.stdout
                sys.stdout = open('trash', 'w')
                play(sample, self.fs)
                sys.stdout = save_stdout
            
            self.lastFrame = sample
        return sample
        
#    def soundGenerationForVideoPurpose (self, objPerFrame) :
#        totalDuration = len(objPerFrame) / 24. # 24 frames per second
#        nbSamplesForOneFrame = int(self.fs/24.)
#        result = np.zeros(int(duration*self.fs))

#        for objects in objPerFrame :
#            self.sound2ChanelsWlength(self.data, nbSamplesForOneFrame, self.totalTime)
#            self.totalTime = self.totaltTime + nbSamples
            
        
        
        

