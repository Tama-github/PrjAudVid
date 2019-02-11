import Vid.kanade as k 
from Vid.Aud.soundgenerator import SoundGenerator
from scipy.io.wavfile import write as wavwrite

sound = "ressources/lb_idle.wav"

def runTest():

    dir = 'ressources/'
    files = [dir + 'test1.mpg', dir + 'test2.mpg', dir + 'test3.mpg', dir + 'test4.mpg', dir + 'test5.mpg']
    
    sg = SoundGenerator(sound);
    j = 0
    for i in files :
        objs = k.kanadeHarris(i, sound)
        data, fs = sg.soundGenerationForVideoPurpose(objs)
        wavwrite("soundTest"+str(j)+".wav",fs,data)
        j+=1
        

def runCam():
    k.kanadeHarris('0', sound)


#runCam()
runTest()

