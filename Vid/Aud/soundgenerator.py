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

OBJECT_SIZE_MODIFIER = 10.
SCREEN_SIZE_PART = 10.
OBJECT_LOW_SPEED_MODIFIER = 100.
OBJECT_HIGHT_SPEED_MODIFIER = 0.1


class SoundGenerator :


    def __init__ (self, soundName) :
        fs, data = wavread(soundName)
        self.fs = fs
        self.data = self.scale(data, -0.5, 0.5)
        self.totalTime = 0
        self.t = time.clock()
        self.frameWidth = 0
        self.frameHeight = 0
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

    def filtrePB (self, x, seuil) :
        X = fftp.fft(x)
        
        tmp1 = X[seuil:int(len(X)/2.)]
        tmp2 = X[int(len(X)/2.):len(X)-seuil]
        
        Y = X*0.
        Y[:int(len(Y)/2.)-seuil] = tmp1
        Y[int(len(Y)/2.)+seuil:] = tmp2
        
        y = fftp.ifft(Y).real
        return y


    def filtrePH (self, x, seuil) :
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
        wavwrite(name, fs, getSoundFrom2Chan(left,right).transpose());
    
    def spatialize (self, data, xPlace) :
        l = data
        r = l
        coef = (xPlace*2)/self.frameWidth - 1.
        l,r = self.moduleChannelsAmplitude(l,r,coef)
        return np.array([l,r])
    
    def fromObjSizeSoundModifier (self, size, sig) :
        screenSize = self.frameWidth * self.frameHeight/SCREEN_SIZE_PART
        if screenSize > size :
            res = self.filtrePB (sig, int((size/screenSize)*OBJECT_SIZE_MODIFIER))
        else :
            res = self.filtrePH (sig, int((screenSize/size)*OBJECT_SIZE_MODIFIER))
        return res
        
    
    def fromObjSpeedSoundModifier (self, speed, sig) :
        if speed < 1. :
            res = self.filtrePB (sig, int(speed*OBJECT_LOW_SPEED_MODIFIER))
        else :
            res = self.filtrePH (sig, int(speed*OBJECT_HIGHT_SPEED_MODIFIER))
        return res
    
    def fromObjCenterSoundModifier (self, center, sig) :
        (x,y) = center
        return self.spatialize(sig, x)
        
    
    def genSampleFromObjects (self, objects, time) :
        tmp = np.zeros(int(time*self.fs))
        sample = self.getSoundFrom2Chan(tmp,tmp)
        sig = self.sound2ChanelsWlength(self.data,int(time*self.fs))
        
        for obj in objects :
            tmp = sig
            (box,center,vector) = obj
            
            # Changing the sound pitch according to the object's size
            (xmin,xmax,ymin,ymax) = box
            objectSize = (xmax - xmin) * (ymax - ymin)
            tmp = self.fromObjSizeSoundModifier(objectSize, tmp)
            
            # Changing the sound pitch according to the object's speed
            (u,v) = vector
            speed = math.sqrt(u*u+v*v)
            tmp = self.fromObjSpeedSoundModifier(speed, tmp)
            
            # Changing the sound's amplitude according to the object place
            tmp = self.fromObjCenterSoundModifier(center, tmp)
            
            sample[0] += tmp[0]
            sample[1] += tmp[1]
            
        return sample
        
    def smoothenStart (self, sample, nb) :
        
        for i in range(nb) :
            sample[0][i] = (sample[0][i]+self.lastFrame[0][len(self.lastFrame)-tmp+i])/2.
            sample[1][i] = (sample[1][i]+self.lastFrame[1][len(self.lastFrame)-tmp+i])/2.
        return sample
        
    
    def soundGenerationForFramePurpose (self, objects) :
        
        tmp = time.clock() - self.t
        sample = []
        if tmp >= 1./24 : # for each frame we want the previous sound to be finished before lauching a new one
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
        
        
        
        
    def soundGenerationForVideoPurpose (self, objPerFramePS) :
        objPerFrame, width, height = objPerFramePS
        
        self.frameHeight = height
        self.frameWidth = width
        
        totalDuration = len(objPerFrame) / 24. # 24 frames per second
        oneFrameDuration = 1./24.
        nbSamplesForOneFrame = int (self.fs/24.)
        
        res = np.zeros(int(self.fs*totalDuration))
        res = np.array([res,res])
        
        i = 0
        for objects in objPerFrame :
            self.sound2ChanelsWlength(self.data, nbSamplesForOneFrame)
            sample = self.genSampleFromObjects(objects, oneFrameDuration)

            debut = i
            if len(res[0]) >= nbSamplesForOneFrame+debut :
                fin = debut+nbSamplesForOneFrame
            else :
                fin = len(res[0])
            
            i = fin

            res[0][debut : fin] = sample[0]
            res[1][debut : fin] = sample[1]
            self.totalTime = self.totalTime + oneFrameDuration
            
        
        return res.transpose(), self.fs
        
        

