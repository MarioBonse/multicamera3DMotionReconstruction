from camera import *

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
