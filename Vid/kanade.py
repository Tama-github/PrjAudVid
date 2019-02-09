import numpy as np
import cv2
import sys
import statistics as stat

#TODO find params wich depend of image size
MIN_CLUSTER_LEN = 20
MIN_LEN_VECTOR = 1
MAX_LEN_VECTOR = 200
X_SAMPLING_DEF = 20
Y_SAMPLING_DEF = 20

EPS_ANGLE = 0.4

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


def clusterVectors(vectors):
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
            print("cluster numéro : " + str(cpt))
            print(len(cluster))
            cpt = cpt + 1
            box = computeBox(cluster)
            center = computeCentroid(cluster)
            vector = computeUniqueVector(cluster)
            obj = (box, center, vector)
            objs.append(obj)
    return objs

def drawScene(objs, frame):

    for (box, center, vector) in objs:
        (x, y) = center
        (u, v) = vector
        (minX, maxX, minY, maxY) = box
        cv2.arrowedLine(frame, (int(x), int(y)), (int(x-u), int(y-v)), (0, 0, 255))
        cv2.circle(frame, (int(x), int(y)),5, (0, 255, 0), -1)
        cv2.rectangle(frame, (minX, minY), (maxX, maxY), (255, 0, 0))




def kanadeTest(videoName):
    if (videoName == '0'):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(videoName)


    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(3000,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    p0 = sampleImage(old_gray)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while(1):
        ret,frame = cap.read()
        if (not(ret)):
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Lukas & Kanade calculation
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]

        vectors = []
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            if thresholdVector((c, d), (a,b), MIN_LEN_VECTOR, MAX_LEN_VECTOR):
                #cv2.arrowedLine(frame, (c, d), (a,b), (0, 0, 255))
                point = (c, d)
                vector = (c-a, d-b)
                vectors.append((point, vector))
        objs = clusterVectors(vectors)
        drawScene(objs, frame)

        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        #p0 = good_new.reshape(-1,1,2)
        p0 = sampleImage(old_gray)
        #p0 = p1
    cv2.destroyAllWindows()
    cap.release()
