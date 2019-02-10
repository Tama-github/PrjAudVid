import numpy as np
import cv2
import sys
import statistics as stat
from Aud.soundgenerator import SoundGenerator 

#TODO find params wich depend of image size
MIN_CLUSTER_LEN = 5
MIN_LEN_VECTOR = 2
MAX_LEN_VECTOR = 200
MINIMUM_COVERAGE = 0.5

EPS_ANGLE = 0.4

#a box can have 30% of the image area
MAX_SURFACE_BOX = 0.3

X_SAMPLING_DEF = 10
Y_SAMPLING_DEF = 10

#TODO when we don't detect object, keep last box, center and vector several frames to enhanced visual perceptions of the tracking

#Create interest points on the frame
def sampleImage(frame):
    deltaX = X_SAMPLING_DEF
    deltaY = Y_SAMPLING_DEF
    #img = cv2.imread(frame,0)
    height, width = frame.shape[:2]
    #print("taille de l'image : " + str(height) + "x" + str(width))

    res = []
    for j in range(deltaX, width, deltaX):
        for i in range(deltaY, height, deltaY):
            res.append([[np.float32(j), np.float32(i)]])
    return np.array(res)

#Use to ignore little and great vectors
def thresholdVector(oldPoint, newPoint, thresholdMin, thresholdMax):
    (a, b) = oldPoint
    (c, d) = newPoint
    (e, f) = (a-c, b-d)
    norm = np.linalg.norm([e,f])
    if norm > thresholdMin and norm < thresholdMax:
        return True
    else:
        return False

#ComputeCentroid from cluster of vectors
def computeCentroid(cluster):
    if (len(cluster) == 0):
        return (0.0, 0.0)
    sum = [0, 0]
    cpt = 0.0
    for (point, vector) in cluster:
        (a, b) = point
        sum[0] = sum[0] + a
        sum[1] = sum[1] + b
        cpt = cpt + 1.0
    sum = [sum[0] / cpt, sum[1] / cpt]

    return (sum[0], sum[1])

#Compute rectangles bounding cluster
def computeBox(cluster):

    minX = sys.maxsize
    maxX = -1
    minY = sys.maxsize
    maxY = -1
    for (point, vector) in cluster:
        (a, b) = point
        if (a < minX):
            minX = a
        if (a > maxX):
            maxX = a
        if (b < minY):
            minY = b
        if (b > maxY):
            maxY = b
        
    return (minX, maxX, minY, maxY)

#compute representative vector
def computeUniqueVector(cluster):

    sum = [0.0, 0.0]
    norms = []
    for (point, vector) in cluster:
        (a, b) = vector
        sum[0] = sum[0] + a
        sum[1] = sum[1] + b
        norms.append(np.linalg.norm(sum))
    median = stat.median(norms)
    sum = [(sum[0]/np.linalg.norm(sum))*median, (sum[1]/np.linalg.norm(sum))*median]
    return (sum[0], sum[1])

def calcBoxArea(box):
    (minX, maxX, minY, maxY) = box
    return (maxX - minX) * (maxY - minY)

#return True if is the same object false else
def testBox(oldBox, newBox):

    (oMinX, oMaxX, oMinY, oMaxY) = oldBox
    (nMinX, nMaxX, nMinY, nMaxY) = newBox
    maxBox = (min(oMinX, nMinX), max(oMaxX, nMaxX), min(oMinY, nMinY), max(oMaxY, nMaxY))
    areaMaxBox = calcBoxArea(maxBox);
    minX = min(oMinX, nMinX)
    minY = min(oMinY, nMinY)
    minBox = (0, oMaxX-oMinX+nMaxX-nMinX, 0, max(oMaxY-oMinY,nMaxY-nMinY))
    minArea = calcBoxArea(minBox)#calcBoxArea(oldBox) + calcBoxArea(newBox)
    if not((areaMaxBox - calcBoxArea(oldBox) - calcBoxArea(newBox)) == 0):
        ratio = minArea/(areaMaxBox - calcBoxArea(oldBox) - calcBoxArea(newBox))
    else:
        ratio = 1.0
    #if (areaMaxBox > minArea):
    if (areaMaxBox/2.0 > calcBoxArea(oldBox) or areaMaxBox/2.0 > calcBoxArea(newBox)):
        return False
    else:
        return True

def testVectors(vec1, vec2):
    (u1, v1) = vec1
    (u2, v2) = vec2

    normalizedV = [u1, v1]/np.linalg.norm([u1, v1])
    normalizedV2 = [u2, v2]/np.linalg.norm([u2, v2])
    angle = np.dot(normalizedV, normalizedV2)
    if (angle >= 1.0-0.3 and angle <= 1.0+0.3):
        return True
    else:
        return False

    
def mergeObjects(obj1, obj2):
    (box1, center1, vector1) = obj1
    (box2, center2, vector2) = obj2
    (minX1, maxX1, minY1, maxY1) = box1
    (minX2, maxX2, minY2, maxY2) = box2
    (x1, y1) = center1
    (x2, y2) = center2
    (u1, v1) = vector1
    (u2, v2) = vector2
    box = (min(minX1, minX2), max(maxX1, maxX2), min(minY1, minY2), max(maxY1, maxY2))
    center = ((x1+x2)/2, (y1+y2)/2)
    norm = np.linalg.norm([u1+u2, v1+v2])
    moyNorm = (np.linalg.norm([u1, v1]) + np.linalg.norm([u2, v2]))/2.0;
    vector = (((u1+u2)/norm) * moyNorm, ((v1+v2)/norm) * moyNorm)
    return (box, center, vector)


def mergeClusters(objs):
    if (len(objs) <= 0):
        return objs
    newObjs = []
    for obj in objs:
        mergeObj = obj
        (box, center, vector) = obj
        for obj2 in objs:
            (box2, center2, vector2) = obj2
            if (not(obj == obj2)):
                if (testBox(box, box2) and testVectors(vector, vector2)):
                #if compareObjs(obj, obj2):
                    mergeObj = mergeObjects(mergeObj, obj2)
                    objs.remove(obj2)
        objs.remove(obj)
        newObjs.append(mergeObj)

    return newObjs

#if we consider vertical objects
def testHorizontalObjects(obj, frame):
    height, width = frame.shape[:2]
    (box, center, vector) = obj
    (minX, maxX, minY, maxY) = box
    if ((maxY-minY)/width > 0.35):
        return True
    else:
        return False


#return true if the box is not to large
def testBoxSize(frame, obj):
    height, width = frame.shape[:2]
    totalArea = height*width
    (box, center, vector) = obj
    (minX, maxX, minY, maxY) = box
    boxArea = (maxX-minX) * (maxY - minY)
    if (boxArea/totalArea < MAX_SURFACE_BOX):
        return True
    else:
        return False

def clusterVectors(vectors, frame):
    eps = EPS_ANGLE
    clusters = []
    for (point, vector) in vectors:
        subCluster = []
        #print(vector)
        (a,b) = vector
        normalizedV = [a, b]/np.linalg.norm([a, b])
        for (pointTemp, vectorTemp) in vectors:
            (c, d) = vectorTemp
            normalizedV2 = [c, d]/np.linalg.norm([c, d])
            angle = np.dot(normalizedV, normalizedV2)
            if (angle > 1-eps and angle < 1+eps):
                subCluster.append((pointTemp, vectorTemp))
                vectors.remove((pointTemp, vectorTemp))
        clusters.append(subCluster)
    
    cpt = 1
    objs = []
    for cluster in clusters:
        if (len(cluster) > MIN_CLUSTER_LEN):
            #print("cluster numero : " + str(cpt))
            #print(len(cluster))
            cpt = cpt + 1
            box = computeBox(cluster)
            center = computeCentroid(cluster)
            vector = computeUniqueVector(cluster)
            obj = (box, center, vector)
            if (testBoxSize(frame, obj) and testHorizontalObjects(obj, frame)):
                objs.append(obj)
    return mergeClusters(objs)

        

def drawScene(objs, frame):

    for (box, center, vector) in objs:
        (x, y) = center
        (u, v) = vector
        (minX, maxX, minY, maxY) = box
        cv2.arrowedLine(frame, (int(x), int(y)), (int(x-u), int(y-v)), (0, 0, 255))
        cv2.circle(frame, (int(x), int(y)),5, (0, 255, 0), -1)
        cv2.rectangle(frame, (minX, minY), (maxX, maxY), (255, 0, 0))


def kanadeHarris(videoName, sample):
    if (videoName == '0'):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videoName)

    fps = cap.get(cv2.CAP_PROP_FPS)
    rem = 0
    oldObjs = []

    # Create the sound generator with the sample's path as parameter 
    sg = SoundGenerator(sample)

    # params for harris corner detection
    feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.001,
                       minDistance = 3,
                       blockSize = 7,
                       useHarrisDetector=True, 
                       k = 0.04)
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(3000,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while(1):
        objects = []

        ret,frame = cap.read()
        if (not(ret)):
            break

        #frame = cv2.bilateralFilter(frame, 5, 2, 2)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Give the frame width to the sound generator
        height, width = frame.shape[:2]
        sg.frameWidth = width

        #Lukas & Kanade calculation
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]

        vectors = []
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #cv2.circle(frame, (a, b), 5, (255, 0, 0))
            if thresholdVector((c, d), (a,b), MIN_LEN_VECTOR, MAX_LEN_VECTOR):
                #cv2.arrowedLine(frame, (c, d), (a,b), (0, 0, 255))
                point = (c, d)
                vector = (c-a, d-b)
                vectors.append((point, vector))
                
        objs = clusterVectors(vectors, frame)
        cv2.arrowedLine(frame, (c, d), (a,b), (0, 0, 255))
        
        if (len(objs) == 0 and rem < fps/2.0):
            objs = oldObjs
            rem = rem + 1
        else:
            oldObjs = objs
            rem = 0
        
        objects.append(objs)



        # Play sound
        sg.soundGenerationForFramePurpose(objs)
        
        # Draw Frame
        drawScene(objs, frame)

        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        #p0 = good_new.reshape(-1,1,2)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        #p0 = p1
    cv2.destroyAllWindows()
    cap.release()
    return objects

