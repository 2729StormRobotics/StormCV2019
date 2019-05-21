######################################################################################################################
#
# JeVois Smart Embedded Machine Vision Toolkit - Copyright (C) 2017 by Laurent Itti, the University of Southern
# California (USC), and iLab at USC. See http://iLab.usc.edu and http://jevois.org for information about this project.
#
# This file is part of the JeVois Smart Embedded Machine Vision Toolkit.  This program is free software; you can
# redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software
# Foundation, version 2.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.  You should have received a copy of the GNU General Public License along with this program;
# if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# Contact information: Laurent Itti - 3641 Watt Way, HNB-07A - Los Angeles, CA 90089-2520 - USA.
# Tel: +1 213 740 3527 - itti@pollux.usc.edu - http://iLab.usc.edu - http://jevois.org
######################################################################################################################
        
import libjevois as jevois
import cv2
import numpy as np
import math # for cos, sin, etc

## Modified example of FIRST Robotics image processing pipeline using OpenCV in Python on JeVois
#
# This module is a modified and simplified version of the C++ module \jvmod{FirstVision}. It is available with
# \jvversion{1.6.2} or later.
#
# This module implements a simple color-based object detector using OpenCV in Python. Its main goal is to also
# demonstrate full 6D pose recovery of the detected object, in Python.
#
# FIXME add pictures
#
# This module isolates pixels within a given HSV range (hue, saturation, and value of color pixels), does some cleanups,
# and extracts object contours. It is looking for the 2019 First Robotics Competition Vision Targets of a specific size
# (set by parameters \p owm and \p ohm for object width and height in meters). See screenshots for an example of shape.
# It sends information about detected objects over serial.
#
# This module usually works best with the camera sensor set to manual exposure, manual gain, manual color balance, etc
# so that HSV color values are reliable. See the file \b script.cfg file in this module's directory for an example of
# how to set the camera settings each time this module is loaded.
#
#  Using this module
#  -----------------
#
# Check out [this tutorial](http://jevois.org/tutorials/UserFirstVision.html) first, for the original
# \jvmod{FirstVision} module written in C++ and also check out the doc for the original \jvmod{FirstVision}. Then
# you can just dive in and start editing the python code of \jvmod{FirstPython}.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# Trying it out
# -------------
#
# Edit the module's file at JEVOIS:/modules/JeVois/FirstPython/FirstPython.py and set the parameters \p self.owm and \p
# self.ohm to the distance between the two bottom points and the y displacement between a bottom point and a top point.
# You should also review and edit the other parameters in the module's constructor, such as the range of HSV colors.
#
# FIXME videomappings incorrect
# 
# @displayname FIRST Python
# @videomapping YUYV 640 252 60.0 YUYV 320 240 60.0 JeVois FirstPython
# @videomapping YUYV 320 252 60.0 YUYV 320 240 60.0 JeVois FirstPython
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2018 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class FirstPython:
    # ###################################################################################################

    ## Constructor
    def __init__(self):
        # HSV color range to use:
        #
        # H: 0=red/do not use because of wraparound, 30=yellow, 45=light green, 60=green, 75=green cyan, 90=cyan,
        #      105=light blue, 120=blue, 135=purple, 150=pink, 180=red
        # S: 0 for unsaturated (whitish discolored object) to 255 for fully saturated (solid color)
        # V: 0 for dark to 255 for maximally bright
        self.HSVmin = np.array([45, 60, 60], dtype=np.uint8)
        self.HSVmax = np.array([90, 255, 255], dtype=np.uint8)

        # Measure the object (in meters) and set its size here:
        self.owm = 0.275 # width in meters
        self.ohm = 0.150 # height in meters

        # Camera Values
        self.fov = 1.13446  # field of view in radians
        self.width = 640    # width of resolution
        self.height = 480   # height of resolution

        # Other processing parameters:
        self.epsilon = 0.019               # Shape smoothing factor (higher for smoother)
        self.hullarea = ( 10*20, 300*300 ) # Range of object area (in pixels) to track
        self.hullfill = 50                 # Max fill ratio of the convex hull (percent)
        self.ethresh = 900                 # Shape error threshold (lower is stricter for exact shape)
        self.margin = 5                    # Margin from from frame borders (pixels)
        self.mToFt = 3.28084               # Conversion of Meters to Feet
        self.cameraAngle = 0.401426        # Angle up from the surface of the floor (radians)

        # Averaging variables
        self.tsum = [[0],[0],[0]]
        self.rsum = [[0],[0],[0]]
        self.sumCount = 0

        # Targeting variables
        self.targetRatio = (300.0/275.0)   # Ratio between distance of top points and bottom points between two targets
        self.pxThreshold = 20              # How close the target can be to the edge of the image
        self.percentFill = 0.1             # Relative amount that the U-Shape map will be filled
    
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("FirstPython", 100, jevois.LOG_INFO)

        # CAUTION: The constructor is a time-critical code section. Taking too long here could upset USB timings and/or
        # video capture software running on the host computer. Only init the strict minimum here, and do not use OpenCV,
        # read files, etc
        
    # ###################################################################################################
    ## Load default camera calibration from JeVois share directory
    def loadCameraCalibration(self, w, h):
        cpf = "/jevois/share/camera/calibration{}x{}.yaml".format(w, h)
        fs = cv2.FileStorage(cpf, cv2.FILE_STORAGE_READ)
        if (fs.isOpened()):
            self.camMatrix = fs.getNode("camera_matrix").mat()
            self.distCoeffs = fs.getNode("distortion_coefficients").mat()
            jevois.LINFO("Loaded camera calibration from {}".format(cpf))
        else:
            jevois.LFATAL("Failed to read camera parameters from file [{}]".format(cpf))

    # ###################################################################################################
    ## Detect objects within our HSV range
    def detect(self, imgbgr, outimg = None):
        maxn = 9 # max number of objects we will consider
        h, w, chans = imgbgr.shape

        # Convert input image to HSV:
        imghsv = cv2.cvtColor(imgbgr, cv2.COLOR_BGR2HSV)

        # Isolate pixels inside our desired HSV range:
        imgth = cv2.inRange(imghsv, self.HSVmin, self.HSVmax)
        outstr = "H={}-{} S={}-{} V={}-{} ".format(self.HSVmin[0], self.HSVmax[0], self.HSVmin[1],
                                                self.HSVmax[1], self.HSVmin[2], self.HSVmax[2])

        # Create structuring elements for morpho maths:
        if not hasattr(self, 'erodeElement'):
            self.erodeElement = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            self.dilateElement = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Apply morphological operations to cleanup the image noise:
        imgth = cv2.erode(imgth, self.erodeElement)
        imgth = cv2.dilate(imgth, self.dilateElement)

        contours, hierarchy = cv2.findContours(imgth, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:maxn]

        # Find and create the raw hulls
        hulls = ()
        centers = ()
        for i in range(len(contours)):
            rawhull = cv2.convexHull(contours[i], clockwise = True)
            rawhullperi = cv2.arcLength(rawhull, closed = True)
            hull = cv2.approxPolyDP(rawhull, epsilon = self.epsilon * rawhullperi * 3.0, closed = True)

            # Outline hull and find center
            huarea = cv2.contourArea(hull, oriented = False)
            if len(hull) == 4 and huarea > self.hullarea[0] and huarea < self.hullarea[1]:
                npHull = np.array(hull, dtype=int).reshape(len(hull),2,1)

                centers += (((npHull[0,0,0] + npHull[1,0,0] + npHull[2,0,0] + npHull[3,0,0]) / 4, (npHull[0,1,0] + npHull[1,1,0] + npHull[2,1,0] + npHull[3,1,0]) / 4),)
                hulls += (npHull,)

        # Reset image
        imgth = cv2.inRange(imghsv, np.array([122, 122, 122], dtype=np.uint8), np.array([122, 122, 122], dtype=np.uint8))

        # Finds the two lowest points of each hull, typically: bottom left point, bottom right point
        bottomPoints = ()
        for i in range(len(hulls)):
            bottomPoint = (-1, 0)
            secondPoint = (-1, 0)
            for k in range(len(hulls[i])):
                if(bottomPoint[0] == -1):
                    bottomPoint = (k, hulls[i][k,1,0])
                elif(bottomPoint[1] < hulls[i][k,1,0]):
                    bottomPoint = (k, hulls[i][k,1,0])

            for k in range(len(hulls[i])):
                if(k != bottomPoint[0]):
                    if(secondPoint[0] == -1):
                        secondPoint = (k, hulls[i][k,1,0])
                    elif(secondPoint[1] < hulls[i][k,1,0]):
                        secondPoint = (k, hulls[i][k,1,0])

            if(abs(centers[i][0] - hulls[i][bottomPoint[0],0,0]) > abs(centers[i][0] - hulls[i][secondPoint[0],0,0])):
                bottomPoints += ((i, bottomPoint[0]),)
            else:
                bottomPoints += ((i, secondPoint[0]),)

        # Find closest other hull to each hull
        nearHull = ()
        for i in range(len(hulls)):
            closest = (-1, 0.0)
            for k in range(len(hulls)):
                if(i != k):
                    if((hulls[bottomPoints[i][0]][bottomPoints[i][1],0,0] - centers[i][0]) * (hulls[bottomPoints[k][0]][bottomPoints[k][1],0,0] - centers[k][0])) < 0:
                        if(centers[i][0] < centers[k][0] and hulls[bottomPoints[i][0]][bottomPoints[i][1],0,0] < centers[i][0]) or (centers[i][0] > centers[k][0] and hulls[bottomPoints[i][0]][bottomPoints[i][1],0,0] > centers[i][0]):
                            if(closest[0] == -1):
                                closest = (k,math.pow(centers[i][0] - centers[k][0], 2) + math.pow(centers[i][1] - centers[k][1], 2))
                            elif(closest[1] > (math.pow(centers[i][0] - centers[k][0], 2) + math.pow(centers[i][1] - centers[k][1], 2))):
                                closest = (k,math.pow(centers[i][0] - centers[k][0], 2) + math.pow(centers[i][1] - centers[k][1], 2))

            nearHull += (closest[0],)

        # Find the two closest points between a hull and its nearest hull
        closePoint = ()
        for i in range(len(hulls)):
            closest = (-1, -1, 0.0)
            for k in range(len(hulls[i])):
                for j in range(len(hulls[nearHull[i]])):
                    if(closest[0] == -1 and closest[1] == -1):
                        closest = (k, j, math.pow(hulls[i][k,0,0] - hulls[nearHull[i]][j,0,0],2) + math.pow(hulls[i][k,1,0] - hulls[nearHull[i]][j,1,0],2))
                    elif(closest[2] > math.pow(hulls[i][k,0,0] - hulls[nearHull[i]][j,0,0],2) + math.pow(hulls[i][k,1,0] - hulls[nearHull[i]][j,1,0],2)):
                        closest = (k, j, math.pow(hulls[i][k,0,0] - hulls[nearHull[i]][j,0,0],2) + math.pow(hulls[i][k,1,0] - hulls[nearHull[i]][j,1,0],2))
            closePoint += ((closest[0], closest[1]),)

        # Find the target center
        hullCenter = []
        for i in range(len(hulls)):
            if(centers[i][0] > centers[nearHull[i]][0] and nearHull[nearHull[i]] == i):
                hullCenter += [(centers[i][0] + centers[nearHull[i]][0]) / 2, (centers[i][1] + centers[nearHull[i]][1]) / 2, i],

        # Choose target closest to center of screen
        targetHull = (-1, 0)
        for i in range(len(hullCenter)):
            if(targetHull[0] == -1):
                targetHull = (hullCenter[i][2], abs(hullCenter[i][0] - self.width / 4))
            elif(targetHull[1] > abs(hullCenter[i][0] - self.width / 4)):
                targetHull = (hullCenter[i][2], abs(hullCenter[i][0] - self.width / 4))

        # Generate the U-Shape map of the target for pose estimation if a target exists
        hlist = []
        if(targetHull[0] != -1):
            # Calculates the displacement of the edge of the physical target to the corner of the drawn target
            xChange = hulls[nearHull[targetHull[0]]][(closePoint[targetHull[0]][1] + 3) % 4,0,0] - hulls[targetHull[0]][(closePoint[targetHull[0]][0] + 1) % 4,0,0]
            yChange = hulls[nearHull[targetHull[0]]][(closePoint[targetHull[0]][1] + 3) % 4,1,0] - hulls[targetHull[0]][(closePoint[targetHull[0]][0] + 1) % 4,1,0]

            # Map points to desired U-Shape
            #
            #   0               3
            #   |               |
            #   |               |
            #   |               |
            #   1_______________2
            #
            corners = (
                (hulls[nearHull[targetHull[0]]][(closePoint[targetHull[0]][1] + 1) % 4,0,0], hulls[nearHull[targetHull[0]]][(closePoint[targetHull[0]][1] + 1) % 4,1,0]),
                (hulls[targetHull[0]][(closePoint[targetHull[0]][0] + 1) % 4,0,0] + xChange*self.targetRatio, hulls[targetHull[0]][(closePoint[targetHull[0]][0] + 1) % 4,1,0] + yChange*self.targetRatio),
                (hulls[nearHull[targetHull[0]]][(closePoint[targetHull[0]][1] + 3) % 4,0,0] - xChange*self.targetRatio, hulls[nearHull[targetHull[0]]][(closePoint[targetHull[0]][1] + 3) % 4,1,0] - yChange*self.targetRatio),
                (hulls[targetHull[0]][(closePoint[targetHull[0]][0] + 3) % 4,0,0], hulls[targetHull[0]][(closePoint[targetHull[0]][0] + 3) % 4,1,0]),
                )

            # Maps U-Shape's corners weighted by Rectangular Corners
            poly = np.array([
                [int(corners[0][0]), int(corners[0][1])],
                [int(corners[1][0]), int(corners[1][1])],
                [int(corners[2][0]), int(corners[2][1])],
                [int(corners[3][0]), int(corners[3][1])],
                [int((corners[3][0] * (1 - self.percentFill) + corners[0][0] * (self.percentFill))), int((corners[3][1] * (1 - self.percentFill) + corners[0][1] * (self.percentFill)))],
                [int((corners[2][0] * (1 - self.percentFill) + corners[0][0] * (self.percentFill))), int((corners[2][1] * (1 - self.percentFill) + corners[0][1] * (self.percentFill)))],
                [int((corners[1][0] * (1 - self.percentFill) + corners[3][0] * (self.percentFill))), int((corners[1][1] * (1 - self.percentFill) + corners[3][1] * (self.percentFill)))],
                [int((corners[0][0] * (1 - self.percentFill) + corners[3][0] * (self.percentFill))), int((corners[0][1] * (1 - self.percentFill) + corners[3][1] * (self.percentFill)))],
                ],np.int32)
    
            # Draw U-Shaped Map
            poly = poly.reshape((-1,1,2))
            cv2.fillPoly(imgth, [poly], (255, 255, 255))

            # Does not pass if the sides of the rectangle are too close to the sides of the image
            isInRange = True
            for point in corners:
                if(point[0] > self.pxThreshold and point[0] < self.width - self.pxThreshold and point[1] > self.pxThreshold and point[1] < self.height - self.pxThreshold): continue
                isInRange = False

            if(isInRange): hlist.append(corners)

        # Display any results requested by the users:
        if outimg is not None and outimg.valid():
            if (outimg.width == w * 2): jevois.pasteGreyToYUYV(imgth, outimg, w, 0)
            jevois.writeText(outimg, "yeet 2.0", 3, h+1, jevois.YUYV.White, jevois.Font.Font6x10)

        # Return the target
        return hlist

    # ###################################################################################################
    ## Estimate 6D pose of each of the quadrilateral objects in hlist:
    def estimatePose(self, hlist):
        rvecs = []
        tvecs = []
        
        # set coordinate system in the middle of the object, with Z pointing out
        # while mapping the rectangular shape of the object
        #
        #   [0]       [3]
        #
        #       (0,0)
        #
        #   [1]       [2]
        #
        objPoints = np.array([ ( -self.owm * 0.5, -self.ohm * 0.5, 0 ),
                               ( -self.owm * 0.5,  self.ohm * 0.5, 0 ),
                               (  self.owm * 0.5,  self.ohm * 0.5, 0 ),
                               (  self.owm * 0.5, -self.ohm * 0.5, 0 ) ])

        # Approximates the pose (position and orientation) of the object
        for detection in hlist:
            det = np.array(detection, dtype=np.float).reshape(4,2,1)
            (ok, rv, tv) = cv2.solvePnP(objPoints, det, self.camMatrix, self.distCoeffs)
            if ok:
                rvecs.append(rv)
                tvecs.append(tv)
            else:
                rvecs.append(np.array([ (0.0), (0.0), (0.0) ]))
                tvecs.append(np.array([ (0.0), (0.0), (0.0) ]))

        # FIXME check table mappings
        # returns pose
        #         returnedArray - definition
        #           [n] - (+)        and  (-)
        #         rvecs - the rotational vector or orientation of target relative to how much it is pointing (+z-axis) at the camera
        #           [0] - right      and  left
        #           [1] - up         and  down
        #           [2] - clockwise  and  counter clockwise
        #         tvecs - the displacement vector of the target out from the camera
        #           [0] - right      and  left
        #           [1] - up         and  down
        #           [2] - forward    and  backward
        return (rvecs, tvecs)        
        
    # ###################################################################################################
    ## Send serial messages, one per object
    def sendAllSerial(self, w, h, hlist, rvecs, tvecs):
        # Initialize by writing the following serial commands:
        # setmapping2 YUYV 640 480 30.0 JeVois FirstPython
        # streamon

        idx = 0
        for c in hlist:
            # Compute quaternion: FIXME need to check!
            tv = tvecs[idx]
            axis = rvecs[idx]
            angle = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) ** 0.5

            # This code lifted from pyquaternion from_axis_angle:
            mag_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]
            if (abs(1.0 - mag_sq) > 1e-12): axis = axis / (mag_sq ** 0.5)
            theta = angle / 2.0
            r = math.cos(theta)
            i = axis * math.sin(theta)
            q = (r, i[0], i[1], i[2])

            # Send x, z displacements and y rotation
            jevois.sendSerial("X: {} Y: {} Angle: {}".
                format(np.asscalar(tv[0]) * self.mToFt, np.asscalar(tv[2]) * self.mToFt, np.asscalar(axis[0])))
            idx += 1
                              
    # ###################################################################################################
    ## Draw all detected objects in 3D
    def drawDetections(self, outimg, hlist, rvecs = None, tvecs = None):
        # Show trihedron and parallelepiped centered on object:
        hw = self.owm * 0.5
        hh = self.ohm * 0.5
        dd = -max(hw, hh)
        i = 0
        empty = np.array([ (0.0), (0.0), (0.0) ])
            
        for obj in hlist:
            # skip those for which solvePnP failed:
            if np.array_equal(rvecs[i], empty):
            	i += 1
            	continue

            jevois.writeText(outimg, str(i), 3, 100 + 10 * i, jevois.YUYV.White, jevois.Font.Font6x10)
            
            # Project axis points:
            axisPoints = np.array([ (0.0, 0.0, 0.0), (hw, 0.0, 0.0), (0.0, hh, 0.0), (0.0, 0.0, dd) ])
            imagePoints, jac = cv2.projectPoints(axisPoints, rvecs[i], tvecs[i], self.camMatrix, self.distCoeffs)
            
            # Draw axis lines:
            jevois.drawLine(outimg, int(imagePoints[0][0,0] + 0.5), int(imagePoints[0][0,1] + 0.5),
                            int(imagePoints[1][0,0] + 0.5), int(imagePoints[1][0,1] + 0.5),
                            2, jevois.YUYV.MedPurple)
            jevois.drawLine(outimg, int(imagePoints[0][0,0] + 0.5), int(imagePoints[0][0,1] + 0.5),
                            int(imagePoints[2][0,0] + 0.5), int(imagePoints[2][0,1] + 0.5),
                            2, jevois.YUYV.MedGreen)
            # Normal to plane
            jevois.drawLine(outimg, int(imagePoints[0][0,0] + 0.5), int(imagePoints[0][0,1] + 0.5),
                            int(imagePoints[3][0,0] + 0.5), int(imagePoints[3][0,1] + 0.5),
                            2, jevois.YUYV.MedGrey)
          
            # Also draw a parallelepiped:
            cubePoints = np.array([ (-hw, -hh, 0.0), (hw, -hh, 0.0), (hw, hh, 0.0), (-hw, hh, 0.0),
                                    (-hw, -hh, dd), (hw, -hh, dd), (hw, hh, dd), (-hw, hh, dd) ])
            cu, jac2 = cv2.projectPoints(cubePoints, rvecs[i], tvecs[i], self.camMatrix, self.distCoeffs)

            # Round all the coordinates and cast to int for drawing:
            cu = np.rint(cu)
          
            # Draw parallelepiped lines:
            jevois.drawLine(outimg, int(cu[0][0,0]), int(cu[0][0,1]), int(cu[1][0,0]), int(cu[1][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[1][0,0]), int(cu[1][0,1]), int(cu[2][0,0]), int(cu[2][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[2][0,0]), int(cu[2][0,1]), int(cu[3][0,0]), int(cu[3][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[3][0,0]), int(cu[3][0,1]), int(cu[0][0,0]), int(cu[0][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[4][0,0]), int(cu[4][0,1]), int(cu[5][0,0]), int(cu[5][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[5][0,0]), int(cu[5][0,1]), int(cu[6][0,0]), int(cu[6][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[6][0,0]), int(cu[6][0,1]), int(cu[7][0,0]), int(cu[7][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[7][0,0]), int(cu[7][0,1]), int(cu[4][0,0]), int(cu[4][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[0][0,0]), int(cu[0][0,1]), int(cu[4][0,0]), int(cu[4][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[1][0,0]), int(cu[1][0,1]), int(cu[5][0,0]), int(cu[5][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[2][0,0]), int(cu[2][0,1]), int(cu[6][0,0]), int(cu[6][0,1]),
                            1, jevois.YUYV.LightGreen)
            jevois.drawLine(outimg, int(cu[3][0,0]), int(cu[3][0,1]), int(cu[7][0,0]), int(cu[7][0,1]),
                            1, jevois.YUYV.LightGreen)

            i += 1

    # ###################################################################################################
    ## Process function with no USB output
    def processNoUSB(self, inframe):
        # Test Serial Output
        #jevois.sendSerial("{} {}".
        #        format(-1 * self.mToFt, -1 * self.mToFt))

        # Get the next camera image (may block until it is captured) as OpenCV BGR:
        imgbgr = inframe.getCvBGR()
        h, w, chans = imgbgr.shape
        
        # Start measuring image processing time:
        self.timer.start()

        # Get a list of quadrilateral convex hulls for all good objects:
        hlist = self.detect(imgbgr)

        # Load camera calibration if needed:
        if not hasattr(self, 'camMatrix'): self.loadCameraCalibration(w, h)

        # Map to 6D (inverse perspective):
        (rvecs, tvecs) = self.estimatePose(hlist)

        # Send all serial messages:
        self.sendAllSerial(w, h, hlist, rvecs, tvecs)

        # Log frames/s info (will go to serlog serial port, default is None):
        self.timer.stop()

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured). To avoid wasting much time assembling a composite
        # output image with multiple panels by concatenating numpy arrays, in this module we use raw YUYV images and
        # fast paste and draw operations provided by JeVois on those images:
        inimg = inframe.get()

        # Start measuring image processing time:
        self.timer.start()
        
        # Convert input image to BGR24:
        imgbgr = jevois.convertToCvBGR(inimg)
        h, w, chans = imgbgr.shape

        # Get pre-allocated but blank output image which we will send over USB:
        outimg = outframe.get()
        outimg.require("output", w * 2, h + 12, jevois.V4L2_PIX_FMT_YUYV)
        #outimg.require("output", w, h + 12, jevois.V4L2_PIX_FMT_YUYV)
        jevois.paste(inimg, outimg, 0, 0)
        jevois.drawFilledRect(outimg, 0, h, outimg.width, outimg.height-h, jevois.YUYV.Black)
        
        # Let camera know we are done using the input image:
        inframe.done()
        
        # Get a list of quadrilateral convex hulls for all good objects:
        hlist = self.detect(imgbgr, outimg)

        # Load camera calibration if needed:
        if not hasattr(self, 'camMatrix'): self.loadCameraCalibration(w, h)

        # Map to 6D (inverse perspective):
        (rvecs, tvecs) = self.estimatePose(hlist)

        # Average Values
        for i in range(len(tvecs)):
            for k in range(len(tvecs[i])):
                self.tsum[k].append(tvecs[i][k])
                while(len(self.tsum[k]) > 10):
                    self.tsum[k].pop(0)

        for i in range(len(rvecs)):
            for k in range(len(rvecs[i])):
                self.rsum[k].append(rvecs[i][k])
                while(len(self.rsum[k]) > 10):
                    self.rsum[k].pop(0)

        # Find distance along ground to robot (Y)
        try:
            X = sum(self.tsum[0]) / len(self.tsum[0]) * self.mToFt
            Y = sum(self.tsum[2]) / len(self.tsum[2]) * self.mToFt
            Z = sum(self.tsum[1]) / len(self.tsum[1]) * self.mToFt
    
            groundDis = -0.2509 + 1.2073 * math.cos(self.cameraAngle - math.atan(Z/Y)) * math.sqrt(math.pow(Z, 2) + math.pow(Y, 2))
        except:
            groundDis = 0
            X = 0
            Y = 0
            Z = 0

        # Output Average of Target in Feet and Radians
        jevois.writeText(outimg, "X: {} Y: {} Angle: {}".format(X, groundDis, (180 / 3.14) * sum(self.rsum[1]) / len(self.rsum[1])), 3, 0, jevois.YUYV.White, jevois.Font.Font6x10)

        # Send all serial messages:
        self.sendAllSerial(w, h, hlist, rvecs, tvecs)

        # Draw all detections in 3D:
        self.drawDetections(outimg, hlist, rvecs, tvecs)

        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        fps = self.timer.stop()
        jevois.writeText(outimg, fps, 3, h-10, jevois.YUYV.White, jevois.Font.Font6x10)

        # Test Serial Output
        #jevois.sendSerial("{} {}".
        #        format(-1 * self.mToFt, -1 * self.mToFt))

        # We are done with the output, ready to send it to host over USB:
        outframe.send()
