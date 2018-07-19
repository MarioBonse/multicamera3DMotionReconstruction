import camera as cam
import 3dplot

def main():
    camera1 = cam.camera(1, "/dev/video0")
    camera2 = cam.camera(2, "/dev/video3")
    if camera1.createCameraMatrixUndistort() == False:
        return False
    if camera2.createCameraMatrixUndistort() == False:
        return False
    camera1.showvideo()
    camera2.showvideo()
    x,y,z = camera1.TriangulateBallVideo(camera2, False)
    3dplot.displayanimation(x, y, z)

if __name__ == "__main__":
    main()
