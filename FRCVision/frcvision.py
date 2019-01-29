import libjevois as jevois
import cv2
import numpy as np
import math
from enum import Enum
from operator import attrgetter

## Simple example of image processing using OpenCV in Python on JeVois
#
# This module is here for you to experiment with Python OpenCV on JeVois.
#
# By default, we get the next video frame from the camera as an OpenCV BGR (color) image named 'inimg'.
# We then apply some image processing to it to create an output BGR image named 'outimg'.
# We finally add some text drawings to outimg and send it to host over USB.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Laurent Itti
# 
# @videomapping YUYV 352 288 30.0 YUYV 352 288 30.0 JeVois FRCVision
# @email itti\@usc.edu
# @address University of Southern California, HNB-07A, 3641 Watt Way, Los Angeles, CA 90089-2520, USA
# @copyright Copyright (C) 2017 by Laurent Itti, iLab and the University of Southern California
# @mainurl http://jevois.org
# @supporturl http://jevois.org/doc
# @otherurl http://iLab.usc.edu
# @license GPL v3
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class FRCVision:
    # ###################################################################################################
    class Point:
        def __init__(self, xVal, yVal):
            self.x = xVal
            self.y = yVal

    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)

        self.outheight = 480
        self.outwidth = 640

        self.__hsv_threshold_hue = [55.505714854722264, 108.89977776139357]
        self.__hsv_threshold_saturation = [0.0, 255.0]
        self.__hsv_threshold_value = [91.72661870503595, 255.0]

        self.hsv_threshold_output = None

        self.__cv_erode_src = self.hsv_threshold_output
        self.__cv_erode_kernel = None
        self.__cv_erode_anchor = (-1, -1)
        self.__cv_erode_iterations = 1.0
        self.__cv_erode_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_bordervalue = (-1)

        self.cv_erode_output = None


        self.__mask_mask = self.cv_erode_output

        self.mask_output = None

        self.__find_lines_input = self.mask_output

        self.find_lines_output = None

    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        inimg = inframe.getCvBGR()
        
        # Start measuring image processing time (NOTE: does not account for input conversion time):
        self.timer.start()

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = inimg
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_erode0:
        self.__cv_erode_src = self.hsv_threshold_output
        (self.cv_erode_output) = self.__cv_erode(self.__cv_erode_src, self.__cv_erode_kernel, self.__cv_erode_anchor, self.__cv_erode_iterations, self.__cv_erode_bordertype, self.__cv_erode_bordervalue)

        # Step Mask0:
        self.__mask_input = inimg
        self.__mask_mask = self.cv_erode_output
        (self.mask_output) = self.__mask(self.__mask_input, self.__mask_mask)

        # Step Find_Lines0:
        self.__find_lines_input = self.mask_output
        (self.find_lines_output) = self.__find_lines(self.__find_lines_input)


        outimg = self.mask_output
        if (len(self.find_lines_output) != 0):
            lines = self.find_lines_output
            points = []
            minGap = 25

            for line in lines:
                # Draws the line
                cv2.line(outimg, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 1)

                # Adds the line's points to an array
                tmp = FRCVision.Point(line.x1, line.y1)
                points.append(tmp)
                tmp = FRCVision.Point(line.x2, line.y2)
                points.append(tmp)

            # Sorts an array twice by x values and y values respectively
            xSort = sorted(points, key=attrgetter('x'))
            ySort = sorted(points, key=attrgetter('y'))

            # Draws a line from the lowest x value to the greatest x value, and a line from the lowest y value to the greatest y value
            #cv2.line(outimg, (ySort[0].x, ySort[0].y), (ySort[len(ySort) - 1].x, ySort[len(ySort) - 1].y), (0, 0, 255), 2)
            #cv2.line(outimg, (xSort[0].x, xSort[0].y), (xSort[len(ySort) - 1].x, xSort[len(ySort) - 1].y), (0, 255, 0), 2)

            # Draws a horizontal line depicting the gaps between areas of interest
            xGap = [xSort[0]]
            for i in range(0, len(xSort) - 2):
                if(abs(xSort[i].x - xSort[i + 1].x) >= minGap):
                    xGap.append(xSort[i])
                    xGap.append(xSort[i + 1])
            xGap.append(xSort[len(xSort) - 1])
            if len(xGap) != 0:
                cv2.line(outimg, (xSort[0].x, math.floor(self.outheight / 2)), (xGap[0].x, math.floor(self.outheight / 2)), (0, 255, 0), 2)
                cv2.line(outimg, (0, math.floor(self.outheight / 2)), (xSort[0].x, math.floor(self.outheight / 2)), (0, 0, 255), 2)
                cv2.line(outimg, (xGap[len(xGap) - 1].x, math.floor(self.outheight / 2)), (xSort[len(xSort) - 1].x, math.floor(self.outheight / 2)), (0, 255, 0), 2)
                cv2.line(outimg, (xSort[len(xSort) - 1].x, math.floor(self.outheight / 2)), (self.outwidth, math.floor(self.outheight / 2)), (0, 0, 255), 2)

                for i in range (0, len(xGap) - 1):
                    if i % 2 == 1:
                        cv2.line(outimg, (xGap[i].x, math.floor(self.outheight / 2)), (xGap[i + 1].x, math.floor(self.outheight / 2)), (0, 0, 255), 2)
                    else:
                        cv2.line(outimg, (xGap[i].x, math.floor(self.outheight / 2)), (xGap[i + 1].x, math.floor(self.outheight / 2)), (0, 255, 0), 2)
                        leftCorner = xGap[i]
                        rightCorner = xGap[i + 1]
                        topCorner = FRCVision.Point(0, 0)
                        bottomCorner = FRCVision.Point(self.outwidth, self.outheight)

                        cv2.line(outimg, (leftCorner.x, leftCorner.y), (rightCorner.x, rightCorner.y), (122, 0, 255), 2)

                        for k in range (0, len(ySort) - 1):
                            if(ySort[len(ySort) - 1 - k].x > leftCorner.x and ySort[len(ySort) - 1 - k].x < rightCorner.x):
                                bottomCorner = ySort[len(ySort) - 1 - k]
                                #cv2.line(outimg, (math.floor(bottomCorner.x - 1), bottomCorner.y), (math.floor(bottomCorner.x + 1), bottomCorner.y), (255, 0, 255), 2)
                                break

                        for k in range (0, len(ySort) - 1):
                            if(ySort[k].x > leftCorner.x and ySort[k].x < rightCorner.x):
                                topCorner = ySort[k]
                                #cv2.line(outimg, (math.floor(topCorner.x - 1), topCorner.y), (math.floor(topCorner.x + 1), topCorner.y), (255, 0, 255), 2)
                                break

                        cv2.line(outimg, (topCorner.x, topCorner.y), (bottomCorner.x, bottomCorner.y), (122, 0, 255), 2)

            else:
                cv2.line(outimg, (xSort[0].x, math.floor(self.outheight / 2)), (xSort[len(xSort) - 1].x, math.floor(self.outheight / 2)), (0, 255, 0), 2)
                cv2.line(outimg, (0, math.floor(self.outheight / 2)), (xSort[0].x, math.floor(self.outheight / 2)), (0, 0, 255), 2)
                cv2.line(outimg, (xSort[len(xSort) - 1].x, math.floor(self.outheight / 2)), (self.outwidth, math.floor(self.outheight / 2)), (0, 0, 255), 2)
        # and so on. When they do "img = cv2.imread('name.jpg', 0)" in these tutorials, the last 0 means they want a
        # gray image, so you should use getCvGRAY() above in these cases. When they do not specify a final 0 in imread()
        # then usually they assume color and you should use getCvBGR() here.
        #
        # The simplest you could try is:
        #    outimg = inimg
        # which will make a simple copy of the input image to output.
        # outimg = cv2.Laplacian(inimg, -1, ksize=5, scale=0.25, delta=127)
                
        # Write a title:
        #cv2.putText(outimg, "JeVois Python Sandbox", (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        # Write frames/s info from our timer into the edge map (NOTE: does not account for output conversion time):
        #fps = self.timer.stop()
        #cv2.putText(outimg, fps, (3, outheight - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        #jevois.writeText(outimg, fps, 3, outheight - 6, jevois.YUYV.White, jevois.Font.Font10x20)


        # Convert our OpenCv output image to video output format and send to host over USB:
        outframe.sendCv(outimg)
        #outframe.send()

    #Grip Methods
    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __cv_erode(src, kernel, anchor, iterations, border_type, border_value):
        """Expands area of lower value in an image.
        Args:
           src: A numpy.ndarray.
           kernel: The kernel for erosion. A numpy.ndarray.
           iterations: the number of times to erode.
           border_type: Opencv enum that represents a border type.
           border_value: value to be used for a constant border.
        Returns:
            A numpy.ndarray after erosion.
        """
        return cv2.erode(src, kernel, anchor, iterations = (int) (iterations +0.5),
                            borderType = border_type, borderValue = border_value)

    @staticmethod
    def __mask(input, mask):
        """Filter out an area of an image using a binary mask.
        Args:
            input: A three channel numpy.ndarray.
            mask: A black and white numpy.ndarray.
        Returns:
            A three channel numpy.ndarray.
        """
        return cv2.bitwise_and(input, input, mask=mask)

    class Line:

        def __init__(self, x1, y1, x2, y2):
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2

        def length(self):
            return numpy.sqrt(pow(self.x2 - self.x1, 2) + pow(self.y2 - self.y1, 2))

        def angle(self):
            return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
    @staticmethod
    def __find_lines(input):
        """Finds all line segments in an image.
        Args:
            input: A numpy.ndarray.
        Returns:
            A filtered list of Lines.
        """
        detector = cv2.createLineSegmentDetector()
        if (len(input.shape) == 2 or input.shape[2] == 1):
            lines = detector.detect(input)
        else:
            tmp = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
            lines = detector.detect(tmp)
        output = []

        # PROBLEM AREA
        if (lines != None and lines[0] != None):
            for i in range(1, len(lines[0])):
                tmp = FRCVision.Line(lines[0][i, 0][0], lines[0][i, 0][1],
                                lines[0][i, 0][2], lines[0][i, 0][3])
                output.append(tmp)
        return output
        #return lines