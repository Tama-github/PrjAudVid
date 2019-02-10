import Vid.kanade as k 

sound = "ressources/lb_idle.wav"

def runTest():

    dir = 'ressources/'
    files = [dir + 'test1.mpg', dir + 'test2.mpg', dir + 'test3.mpg', dir + 'test4.mpg', dir + 'test5.mpg']
    

    for i in files:
        k.kanadeHarris(i, sound)

def runCam():
    k.kanadeHarris('0', sound)


#runCam()
runTest()

