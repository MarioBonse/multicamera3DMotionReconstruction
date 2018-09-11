class HSVObjTracking():
    """docstring for ."""
    def __init__(self, lowerBoung, upperBoung, cameras):
        self.camera = []
        self.cameranumber = len(cameras)
        if self.cameranumber < 2:
            print("Error, impossible to track object without at least 2 cameras")
            return False
        for i in cameras:
            if camera[i]. undistorted == False OR camera[i].calibrated == False:
                print("Camera with id +"camera[i].id+"is not well calibrated")
                self.camera = []
                return False
            self.camera.append(i)
        self.lowerBound = lowerBound
        self.higherBoung = upperBoung
        self.ready = True
        return True

    def findcenter(self, image):
        ballLower = self.lowerBound
        ballUpper = self.upperBound
        blurred = cv2.GaussianBlur(image,(5,5),0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #sottraggo dallo sfondo il colore(nel mio caso giallo)
        mask = cv2.inRange(hsv, ballLower, ballUpper)
        #con erode e dilate elimno rumore
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        im2, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center = None
        tuple = None
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            #to find the center
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = np.matrix([[int(M["m10"] / M["m00"])],[int(M["m01"] / M["m00"])]])
            tuple = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        return center, tuple

    def threedMovementsRecontstruction(self):
        buffersize = 32
        video = False
        frameList = []
        obj = []
        counter = range(self.cameranumber)
        pts = collections.deque(maxlen = buffersize)
        cap = []
        pose = False
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xvector = np.array([])
        yvector = np.array([])
        zvector = np.array([])
        for i in self.camera:
            cap.append(cv2.VideoCapture(i.device))
        if video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output_rimbalzo.avi',fourcc, 20.0, (int(cap[0].get(3)), int(cap[0].get(4))))
        time.sleep(2.0)
        for i in range(1, 500):
            cont = False
            #with the first frame we try to find camera pose
            for i in counter:
                ret, frame = cap[i].read
                framelist[i] = frame
                if ret == False
                    break
            frame = frame[0].copy()
            for i in counter:
                if self.camera[i].poseEstimate(frame[i]) == False
                    print("Impossibile trovare la posizione della palla nello spazio -> fallito calcolo posa")
                    return False
            pose = True
            for i in counter:
                center[i], centerTuple[i] = findCenter(frame[i])
                if center[i] is None:
                    cont = True
            if cont = True:
                continue

            #code for representing track on a camera(usually camera 0)
            pts.appendleft(centreTuple[0])
            # loop over the set of tracked points
            for i in range(1, len(pts)):
                #if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(buffersize / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            # show the frame to our screen
            cv2.imshow("Frame", frame)
            if video:
                out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            for i in counter:
                obj[i] = [camera[i].projectionMatrix, center[i]]

            x3d = camera1.findPoint(obj)
            xvector = np.append(xvector, [x3d[0]])
            yvector = np.append(yvector, [x3d[1]])
            zvector = np.append(zvector, [x3d[2]])
        for i in range:
            cap[i].release()
        if video:
            out.release()
        cv2.destroyAllWindows()
        ax.plot3D(xvector, yvector, zvector, 'gray')
        return(xvector, yvector, zvector)
