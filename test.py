from Vid.Aud.soundgenerator import SoundGenerator 
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

sound = SoundGenerator("ressources/lb_idle.wav")

fs,s = wavread("ressources/lb_idle.wav")

s = sound.scale(s,-1.,1)
s = sound.sound2ChanelsWlength(s,fs)

play(s,fs)

