import Vid.kanade as k 
from Vid.Aud.soundgenerator import SoundGenerator
from scipy.io.wavfile import write as wavwrite
import os

sound = "ressources/lb_idle.wav"


def genNewVideo(videoName, soundName, ind):
    cmd = 'ffmpeg -i ' + videoName + ' -i ' + soundName + ' -map 0:0 -map 1:0 -c:v copy -c:a copy ' + 'res/output' + str(ind) + '.avi'
    os.system(cmd)

    #ffmpeg -i <sourceVideoFile> -i <sourceAudioFile> -map 0:0 -map 1:0 -c:v copy -c:a copy <outputVideoFile>

def concatRes(fileList):
    cmd = 'ffmpeg -i "concat:'
    for file in fileList:
        cmd = cmd + file + '|'
    cmd = cmd + '" -codec copy res/output.avi'
    print(cmd)
    os.system(cmd)

    #ffmpeg -i "concat:res/output0.avi|res/output1.avi" -codec copy output.avi


def runTest():

    dir = 'ressources/'
    files = [dir + 'test1.mpg', dir + 'test2.mpg', dir + 'test3.mpg', dir + 'test4.mpg', dir + 'test5.mpg']
    
    sg = SoundGenerator(sound);
    j = 0
    fileList = []
    for i in files :
        objs = k.kanadeHarris(i, sound)
        data, fs = sg.soundGenerationForVideoPurpose(objs)
        wavwrite("soundTest"+str(j)+".wav",fs,data)
        genNewVideo('ressources/test'+str(j+1)+'.mpg_out.avi', "soundTest"+str(j)+".wav", j)
        fileList.append('res/output' + str(j) + '.avi')
        j+=1
    concatRes(fileList)

def runCam():
    k.kanadeHarris('0', sound)


#runCam()
runTest()

