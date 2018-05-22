
import numpy as np
import cv2
import glob
import pickle
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#chessboard dimention
chessboard_row = 9
chessboard_col = 6

class camera:
    undistorted = False
    poseEstimated = False
    mtx = 0
    dist = 0
    rotationVector = 0
    rotationMatrix = 0
    RTmatrix = 0
    tvecs = 0

    def __init__(self, id):
        self.camera_number = str(id)


    def createCameraMatrixUndistort(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_row * chessboard_col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_col, 0:chessboard_row].T.reshape(-1, 2)


        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob("camera"+ self.camera_number+"/*.jpg")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_col, chessboard_row), None)


            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                #cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9,6), corners, True)
                cv2.imshow('img',img)
                cv2.waitKey(1) & 0xff
        cv2.destroyAllWindows()

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        #save in file
        self.undistorted = True
        self.saveFile()

    def saveFile(self):
        if self.undistorted == False:
            return False
        else:
            pickle.dump( [self.mtx, self.dist], open( "cameraCoefficient_"
                                                    + self.camera_number + ".p", "wb" ) )

    def loadCoefficient(self):
        if self.undistorted == True:
            return(True)
        try:
            with open("cameraCoefficient_" + self.camera_number + ".p", "rb") as f:
                self.mtx, self.dist = pickle.load(f)
            self.undistorted = True
            return(True)
        except:
            return(False)


    def undistort(self, img):
        if self.undistorted == True:
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
            undistorted_img = cv2.undistort(img, self.mtx, aelf.dist, None, newcameramtx)
            x,y,w,h = roi
            undistorted_img = undistort_img[y:y+h, x:x+w]
            return(undistorted_img)
        return False

    def poseEstimate(camera):
        if camera.undistorted == False:
            return False
        else:
            foto_path = 'pose_camera_'+self.camera_number+'.jpg'
            img1 = cv2.imread(foto_path,0)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((chessboard_row * chessboard_col, 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_col, 0:chessboard_row].T.reshape(-1, 2)

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_col, chessboard_row), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                # Find the rotation and translation vectors.
                self.rotationVector, self.tvecs, _ = cv2.solvePnPRansac(objp, corners2, self.mtx, self.dist)
                self.poseEstimated = True
                return True
            else:
                print("Impossibile calcolare la posa della camera dalla seguente foto "+foto_path)
                return False

    def composeProjectionMatrix(self):
        self.rotationMatrix = Rodriguez(rotationVector)
        #
        #write the projection from camera cordinate to chessboard cordinate as
        #      (                     TVEC_A )
        # RT = (    ROTMATRIX        TVEC_B )
        #      (                     TVEC_C )
        #
        #we obtain projection matrix as
        #
        # PJM = M*RT
        #
        self.tvecs = np.reshape(self.tvecs, (3,1))
        self.RTmatrix = np.concatenate((self.rotationMatrix, self.tvecs), axis = 1)
        self.projectionMatrix = np.matmul(self.mtx, self.RTmatrix)


    #function that estimate the position of a point in the space.
    #argouments are as many couople of points in camera cordinate and projection matrix
    #as we can(at least 2).
    # findPoint([projeciton1, point1], [pojection2, point2]...)
    def findPoint(*arg):
        #pseudoinvesa delle projection matrix(P)
        #concatenate per colonne(incolonnate)
        #X = P*x
        #con x insieme dei punti in cordinate di camer corrispondenti
        n_points = len(arg)
        if n_points<2:
            print("Numero di punti insufficiente")
            return False
        if chechArgument(arg[0]):
            return False
        A = arg[0][0]
        x = arg[0][1]
        for i in (1:n_points-1):
            if chechArgument(arg[i]):
                return False
            A = vstack(A, arg[i][0])
            app = vstack(arg[i][1], 1)#scrivo in cordinate omogenee, passo da (x,y) a (x,y,1)
            x = vstack(x, app)
        A_psin = np.linalg.pinv(A)
        X = np.dot(A_psin, x)
        return X


    def chechArgument(arg):
        if shape(arg[0])!= (3,4) OR shape(arg[1]) != (2,1):#projection matrix must be (3x4)
            print("Bad argument")
            return False
        return True



    #non lo uso più
    def FMatrixFromImages(camera1, camera2):
        img1 = cv2.imread('c1.jpg',0)   # left image
        img2 = cv2.imread('c2.jpg',0)  # right image
        camera1.loadCoefficient()
        camera2.loadCoefficient()
        img1 = camera1.undistort(img1)
        img2 = camera1.undistort(img1)
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good = []
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        return(F)



def main():
    camera0 = camera(0)
    camera1 = camera(1)
    #try to load the camera matrix and other coefficient from the file.
    #If the file hasn't been written we try to create the parameters and then save them.
    camera0.loadCoefficient()
    camera0.undistorted
    camera1.loadCoefficient()
    if camera0.undistorted == False:
        try:
            camera0.createCoefficient()
        except:
            print("Error, camera images not present for camera number "+camera0.camera_number)
    if camera1.undistorted == False:
        try:
            camera1.createCoefficient()
        except:
            print("Error, camera images not present for camera number "+camera1.camera_number)
    camera0.poseEstimate()
    camera0.composeProjectionMatrix()

    camera1.poseEstimate()
    camera1.composeProjectionMatrix()

main()
