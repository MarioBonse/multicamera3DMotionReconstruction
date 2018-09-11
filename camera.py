"""
#this file contain camera class with function for:
#Undistort camera
#Estimate Camera Pose
#Estimate point position in the space
"""
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#chessboard dimention
chessboard_row = 6
chessboard_col = 9




class camera:
    undistorted = False
    poseEstimated = False

    def __init__(self, id, name):
        self.camera_number = str(id)
        self.device = name

    """
    Function that recover camera intrinsic parameters and camera Distortion Coefficient.
    For each camera we need a directory with many chessboard phothos for calculating them.
    The name of the directory will be cameraID.
    Ogtherwise there should be a file called cameraCoefficient_ID.p with the coefficient already saved.
    """
    def createCameraMatrixUndistort(self):
        if self.loadCoefficient():
            return true
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_row * chessboard_col, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_row, 0:chessboard_col].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        images = glob.glob("camera"+ self.camera_number+"/*.jpg")

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_row, chessboard_col), None)


            # If found, add object points, image points (after refining them)
            if ret == True:
                corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                objpoints.append(objp)

                #cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9,6), corners, True)
                cv2.imshow('img',img)
                cv2.waitKey(1) & 0xff
        cv2.destroyAllWindows()

        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        if ret == False:
            print("Error, camera images not present for camera number "+camera2.camera_number)
            return False
        #save in file
        self.undistorted = True
        self.saveFile()
        return True

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


    def undistort_frame(self, img):
        if self.undistorted == True:
            h,  w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx,self.dist,(w,h),1,(w,h))
            undistorted_img = cv2.undistort(img, self.mtx, self.dist, None, newcameramtx)
            x,y,w,h = roi
            undistorted_img = undistorted_img[y:y+h, x:x+w]
            return(undistorted_img)
        return False

    def showvideo(self):
        if self.undistorted == False:
            return False
        cap = cv2.VideoCapture(self.device)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == False:
                print("Error: camera "+self.camera_number+" is no attached to "+self.device)
                break
            frame = self.undistort_frame(frame)
            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    def checkrotation(self, corners, photo):
        #black and white photo
        #CONTROLLO RAPPRESENTAZIONE DEI PUNTI
        #passo il vettore dei corner -> il primo elemento è il coner in basso a sinistra il [n_row + 1] è quello
        #sopra a destra rispetto a lui
        #quindi il punto medio x1+x2/2 è il centro della prima casa
        #
        #in modo uguale  chessboard_row*chessboard_col - 1 è l'ultimo
        #mentre quello in basso a sinistra rispetto a lui è
        # (row*col-1)-row-1
        #
        #the first square is black then the chessboard is straight->return False
        #is white then it's rotated -> return true
        lowermedium = (corners[0] + corners[chessboard_row + 1])/2
        uppermedium = (corners[(chessboard_row)*(chessboard_col-1)-2] + corners[chessboard_row*chessboard_col - 1])/2
        lowermedium = lowermedium.astype(int)
        uppermedium = uppermedium.astype(int)
        #if lower corner is black -> chessbord is not rotated
        #if lowercorner < uppercorner then it's black
        if photo[lowermedium[0, 1]][lowermedium[0, 0]] > photo[uppermedium[0, 1]][uppermedium[0, 0]]:
            return True
        return False


    """
    Function that find camera pose given a chessboard photo and a camera.
    """
    def poseEstimate(self, img):
        if self.undistorted == False:
            return False
        else:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chessboard_row, chessboard_col), None)
            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                rotated = self.checkrotation(corners2, gray)
                # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
                objp = np.zeros((chessboard_row * chessboard_col, 3), np.float32)
                objp[:, :2] = np.mgrid[0:chessboard_row, 0:chessboard_col].T.reshape(-1, 2)
                #if chessboard is rotated then we change the axis numeration
                # so object points, like (5,8,0), (4,8,0), (4,8,0) ....,(0,0,0)
                if(rotated):
                    for d in objp:
                        d[0] = (chessboard_row-1)-d[0]
                        d[1] = (chessboard_col - 1) - d[1]
                # Find the rotation and translation vectors.
                _, self.rotationVector, self.tvecs, _ = cv2.solvePnPRansac(objp, corners2, self.mtx, self.dist)
                self.poseEstimated = True
                self.composeProjectionMatrix()
                return True
            else:
                print("Impossibile calcolare la posa della camera errore 1200")
                return False



    """
    #write the projection from camera cordinate to chessboard cordinate as
    #      (                     TVEC_A )
    # RT = (    ROTMATRIX        TVEC_B )
    #      (                     TVEC_C )
    #
    #we obtain projection matrix as
    #
    # PJM = M*RT
    #
    """
    def composeProjectionMatrix(self):
        self.rotationMatrix, _ = cv2.Rodrigues(self.rotationVector)
        self.tvecs = np.reshape(self.tvecs, (3,1))
        self.RTmatrix = np.concatenate((self.rotationMatrix, self.tvecs), axis = 1)
        self.projectionMatrix = np.matmul(self.mtx, self.RTmatrix)


"""
#function that estimate the position of a point in the space.
#argouments are as many couople of points in camera cordinate and projection matrix
#as we can(at least 2).
# findPoint([projeciton1, point1], [pojection2, point2]...)
"""
def findPoint(arg):
    #projectionmatrix = Pi
    #points in camera cordinates xi
    # xi.shape = (3*1)
    #    (P1)          (x1)
    # P =(P2)     x  = (x2)
    #     ..             ..
    #P+ = pseudoinvers(P)
    # X = p+*x
    # X.shape = (4, 1)
    #pseudoinvers of projection matrix(P)
    #concatenate colomn
    #X = P*x
    #con x insieme dei punti in cordinate di camer corrispondenti
    n_points = len(arg)
    if n_points<2:
        print("Numero di punti insufficiente")
        return False
    if arg[0][0].shape != (3,4) or arg[0][1].shape != (2,1):#projection matrix must be (3x4)
        print("Bad argument")
        return False
    A = arg[0][0]
    x =  np.vstack((arg[0][1],1))
    for i in range(1,n_points-1):
        if arg[i][0].shape != (3,4) or arg[i][1].shape != (2,1):#projection matrix must be (3x4)
            print("Bad argument")
            return False
        A = np.vstack((A, arg[i][0]))
        app = np.vstack((x, arg[i][1], 1))#scrivo in cordinate omogenee, passo da (x,y) a (x,y,1)
    A_psin = np.linalg.pinv(A)
    X_3d = np.dot(A_psin, x)
    return X_3d


