import threeDplot
import camera as cam
import HSVObjTracking as HSVTr
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def main():
    ######red ball
    ballLower = (137, 88, 55)
    ballUpper = (183, 255, 255)
    camera1 = cam.camera(1, "/dev/video0")
    camera2 = cam.camera(2, "/dev/video3")
    if camera1.createCameraMatrixUndistort() == False:
        return False
    if camera2.createCameraMatrixUndistort() == False:
        return False
    camera1.showvideo()
    camera2.showvideo()
    trackingObject = HSVTr.HSVObjTracking(ballLower, ballUpper, [camera1, camera2])
    x,y,z = trackingObject.threedMovementsRecontstruction()
    threeDplot.displayanimation(x, y, z)

if __name__ == "__main__":
    main()


###yellow ball
#ballLower = (14, 67, 34)
#ballUpper = (57, 255, 255)
#
######red ball
#ballLower = (137, 88, 55)
#ballUpper = (183, 255, 255)
