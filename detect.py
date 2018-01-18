import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys, os
import traceback
from moviepy.editor import VideoFileClip
from moviepy.editor import TextClip
from moviepy.editor import CompositeVideoClip

global left_lanes, right_lanes, bad_left_lanes, bad_right_lanes
global left_failed_count, right_failed_count
CUMSUM_WINDOW = 6 # Number of lanes to take averages on in the smoothing function

# This class is used to record the characteristics of each lane detection
class Lane():
    def __init__(self):
        #polynomial coefficients for the polyfit function
        self.polycof = [np.array([False])]
        #radius of curvature of the lane in meters
        self.radius_of_curvature = None
        #x values for detected lane pixels
        self.allx = None
        #y values for detected lane pixels
        self.ally = None

### The camera is calibrated here using a series of chessboard images captured by the camera ###
### The calibration results are saved as a pickle file which could be retrieved when this program is called ###

filename = "wide_dist_pickle.p"
cur_dir = os.getcwd()
file_list = os.listdir(cur_dir + "/camera_cal/")

if filename not in file_list:
    # Pickle file not found (i.e. camera not calibrated)

    import glob

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/cali*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Test undistortion on an image
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('camera_cal/test_undist.jpg',dst)

    # Save the camera calibration result for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "camera_cal/" + filename, "wb" ) )

# Retrieve calibration results from the pickle file

try:
    dist_pickle = pickle.load(open("camera_cal/" + filename, "rb"))
except (AttributeError,  EOFError, ImportError, IndexError) as e:
    print(traceback.format_exc(e))
    pass
except Exception as e:
    print(traceback.format_exc(e))
    sys.exit()

mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


### This function creates a thresholded binary image of a raw color image using color transform and ###
### gradient thresholding ###

def pipeline(img, s_thresh=(150, 255), sx_thresh=(30, 100)):
    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    color_binary = color_binary.astype(np.uint8)

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary

###  This function transforms the perspective of an image into a top down view

def warp(img):

    # Define the vertices of a four sided polygon mask.  The vertices locations are
    # derived from an image captured from the front of the car on lane lines that are straight (i.e. not curved)
    imshape = img.shape
    left_bottom = [188, imshape[0]]
    left_top = [590, 450]
    right_top = [690, 450]
    right_bottom = [1120, imshape[0]]

    src = np.float32(
        [left_bottom,
         left_top,
         right_top,
         right_bottom])

    # The pixel area inside the polygon mask is mapped to the following pixel area after the transformation
    dst = np.float32(
           [[200, imshape[0]], [200, 0], [1000, 0], [1000, imshape[0]]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)  # Performs the transformation
    return warped, M, Minv


### This function identifies the left and right lane lines using sliding windows that move vertically
### across the image to track the lane pixels.  The sliding windows move in two directions: top down and bottom up.
### The windows first move down from the top of the image before going back up from the bottom, with both iterations
### halting when they hit the center of the image.  If the windows formed from the top and the bottom join up
### in the middle, then there is a good chance for the lane line to be found. Otherwise, if the windows do not join
### up, then the top down windows will be wiped and the bottom up windows will continue to iterate until it reaches
### the top

def detect_lanes(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram_bot = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Take a histogram of the top one-ninth of the image, which represents the height of a window
    histogram_top = np.sum(binary_warped[:binary_warped.shape[0]/9,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram on the bottom half of the image
    # These will be the starting point for the left and right lines
    midpoint_bot = np.int(histogram_bot.shape[0]/2)
    leftx_base = np.argmax(histogram_bot[:midpoint_bot])
    rightx_base = np.argmax(histogram_bot[midpoint_bot:]) + midpoint_bot

    # Do the same thing for the top half
    midpoint_top = np.int(histogram_top.shape[0] / 2)
    leftx_top = np.argmax(histogram_top[:midpoint_top])
    rightx_top = np.argmax(histogram_top[midpoint_top:]) + midpoint_top

    # Choose the number of sliding windows for the top and bottom half of the image
    nwindows_bot = 5
    nwindows_top = 4
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/(nwindows_bot + nwindows_top))
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current_bot = leftx_base
    rightx_current_bot = rightx_base
    leftx_current_top = leftx_top
    rightx_current_top = rightx_top
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set maximum pixel number for a window
    maxpix = 4000
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Initialize the vertices of the various windows
    win_bot_y_low, win_bot_y_high, win_bot_xleft_low = 0, 0, 0
    win_bot_xleft_high, win_bot_xright_low, win_bot_xright_high = 0, 0, 0
    win_top_y_low, win_top_y_high, win_top_xleft_low = 0, 0, 0
    win_top_xleft_high, win_top_xright_low, win_top_xright_high = 0, 0, 0
    # Count the number of windows that were appended to the left and right lane pixel indices
    # in the top half of the image
    top_left_count, top_right_count = 0, 0

    # Step through the windows one by one from the top of the image
    for window in range(nwindows_top):
        # Identify window boundaries in x and y (and right and left)
        win_top_y_low = window * window_height
        win_top_y_high = (window + 1) * window_height
        win_top_xleft_low = leftx_current_top - margin
        win_top_xleft_high = leftx_current_top + margin
        win_top_xright_low = rightx_current_top - margin
        win_top_xright_high = rightx_current_top + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_top_xleft_low, win_top_y_low), (win_top_xleft_high, win_top_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_top_xright_low, win_top_y_low), (win_top_xright_high, win_top_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_top_y_low) & (nonzeroy < win_top_y_high) &
                          (nonzerox >= win_top_xleft_low) & (nonzerox < win_top_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_top_y_low) & (nonzeroy < win_top_y_high) &
                           (nonzerox >= win_top_xright_low) & (nonzerox < win_top_xright_high)).nonzero()[0]
        # Append these indices to the lists if the min and max pixel conditions are met
        # Max pixel is used to discount those windows with too much noise data
        if minpix < len(good_left_inds) < maxpix:
            left_lane_inds.append(good_left_inds)
            leftx_current_top = np.int(np.mean(nonzerox[good_left_inds]))  # Recenter the next window using the mean
            top_left_count += 1  # If the top windows don't match up with the bottom windows, this helps to track
                                 # the number of windows to remove from the indices later on

        if minpix < len(good_right_inds) < maxpix:
            right_lane_inds.append(good_right_inds)
            rightx_current_top = np.int(np.mean(nonzerox[good_right_inds]))
            top_right_count += 1

        # Print out the number of pixels within each window
        cv2.putText(out_img, str(len(good_left_inds)), (win_top_xleft_low, win_top_y_high), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 0))
        cv2.putText(out_img, str(len(good_right_inds)), (win_top_xright_low, win_top_y_high), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0))

    # Step through the windows one by one on the left side of the bottom half of the image
    for window in range((nwindows_top + nwindows_bot)):
        # Identify window boundaries in x and y (and right and left)
        win_bot_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_bot_y_high = binary_warped.shape[0] - window * window_height
        win_bot_xleft_low = leftx_current_bot - margin
        win_bot_xleft_high = leftx_current_bot + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_bot_xleft_low, win_bot_y_low), (win_bot_xleft_high, win_bot_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_bot_y_low) & (nonzeroy < win_bot_y_high) &
                          (nonzerox >= win_bot_xleft_low) & (nonzerox < win_bot_xleft_high)).nonzero()[0]
        # Append these indices to the list if noise level in window is low
        if len(good_left_inds) < maxpix:
            left_lane_inds.append(good_left_inds)
        # Center the next window using the mean value of the current window
        if minpix < len(good_left_inds):
            leftx_current_bot = np.int(np.mean(nonzerox[good_left_inds]))

        # Print out the number of pixels within each window
        cv2.putText(out_img, str(len(good_left_inds)), (win_bot_xleft_low, win_bot_y_high),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

        # If the windows from the top and bottom halves match up, break the for loop.  Otherwise delete
        # the windows that were captured previously from the top half and continue iterating to the top of the image
        if (window == (nwindows_bot - 1)):
            if (win_top_xleft_low > win_bot_xleft_high or win_top_xleft_high < win_bot_xleft_low):
                del left_lane_inds[:top_left_count]
            else:
                break

    # Step through the windows one by one on the right side of the bottom half of the image
    for window in range((nwindows_top + nwindows_bot)):
        # Identify window boundaries in x and y (and right and left)
        win_bot_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_bot_y_high = binary_warped.shape[0] - window * window_height
        win_bot_xright_low = rightx_current_bot - margin
        win_bot_xright_high = rightx_current_bot + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_bot_xright_low, win_bot_y_low), (win_bot_xright_high, win_bot_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_right_inds = ((nonzeroy >= win_bot_y_low) & (nonzeroy < win_bot_y_high) &
                          (nonzerox >= win_bot_xright_low) & (nonzerox < win_bot_xright_high)).nonzero()[0]
        # Append these indices to the lists
        if len(good_right_inds) < maxpix:
            right_lane_inds.append(good_right_inds)

        if minpix < len(good_right_inds) < maxpix:
            rightx_current_bot = np.int(np.mean(nonzerox[good_right_inds]))

        cv2.putText(out_img, str(len(good_right_inds)), (win_bot_xright_low, win_bot_y_high),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

        # If the windows from the top and bottom halves match up, break the for loop.  Otherwise delete
        # the windows that were captured previously from the top half and continue iterating to the top of the image
        if (window == (nwindows_bot - 1)):
            if (win_top_xright_low > win_bot_xright_high or win_top_xright_high < win_bot_xright_low):
                del right_lane_inds[:top_right_count]
            else:
                break

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_pts = np.concatenate(np.array([np.transpose(np.vstack([left_fitx, ploty]))]))
    right_pts = np.concatenate(np.array([np.transpose(np.vstack([right_fitx, ploty]))]))

    # Draw curves on out_img
    cv2.polylines(out_img, np.int32([left_pts]), 0, (0, 255, 0), 3)
    cv2.polylines(out_img, np.int32([right_pts]), 0, (255, 0, 0), 3)

    # Compute radius of curvature
    left_curverad = cal_curvature(left_fit)
    right_curverad = cal_curvature(right_fit)

    # Record left lane info
    left_lane = Lane()
    left_lane.allx = left_fitx
    left_lane.ally = ploty
    left_lane.polycof = left_fit
    left_lane.radius_of_curvature = left_curverad
    left_lane.windowx = leftx
    left_lane.windowy = lefty

    # Record right lane info
    right_lane = Lane()
    right_lane.allx = right_fitx
    right_lane.ally = ploty
    right_lane.polycof = right_fit
    right_lane.radius_of_curvature = right_curverad
    right_lane.windowx = rightx
    right_lane.windowy = righty

    return left_lane, right_lane, out_img

### This function computes the radius of the curvature of a lane

def cal_curvature(lanefit):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 720

    # scale derivatives using conversion ratios
    dxdy = (2 * lanefit[0] * y_eval * ym_per_pix * xm_per_pix / (ym_per_pix ** 2)) + (
        xm_per_pix / ym_per_pix * lanefit[1])
    dxdy_sq = 2 * lanefit[0] * xm_per_pix / (ym_per_pix ** 2)

    # Calculate curvature using given formula
    return ((1 + (dxdy) ** 2) ** 1.5) / np.absolute(dxdy_sq)

### This function draws the lane lines on a blank image

def draw_lanes(binary_warped):

    global left_lanes, right_lanes

    if len(left_lanes) > 1:
        left_lane = cal_mean_lane(left_lanes, CUMSUM_WINDOW)
    else:
        left_lane = left_lanes[-1]

    if len(right_lanes) > 1:
        right_lane = cal_mean_lane(right_lanes, CUMSUM_WINDOW)
    else:
        right_lane = right_lanes[-1]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.allx, left_lane.ally]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.allx, right_lane.ally])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return color_warp, left_lane, right_lane

### This fucntion compares two lane lines and see if they are roughly parallel and have roughly the same curvature

def sanity_check(oldlane, newlane):

    recent_lane = oldlane
    recent_lane_cof = recent_lane.polycof

    # Check gradients at three points

    recent_lane_dydx_p1 = 2 * recent_lane_cof[0] * recent_lane.ally[0] + recent_lane_cof[1]
    recent_lane_dydx_p2 = 2 * recent_lane_cof[0] * recent_lane.ally[int(len(recent_lane.ally) / 2)] + recent_lane_cof[1]
    recent_lane_dydx_p3 = 2 * recent_lane_cof[0] * recent_lane.ally[-1] + recent_lane_cof[1]

    newlane_dydx_p1 = 2 * newlane.polycof[0] * newlane.ally[0] + newlane.polycof[1]
    newlane_dydx_p2 = 2 * newlane.polycof[0] * newlane.ally[int(len(newlane.ally) / 2)] + newlane.polycof[1]
    newlane_dydx_p3 = 2 * newlane.polycof[0] * newlane.ally[-1] + newlane.polycof[1]

    gradient_threshold = 1

    # True if parallel, otherwise false
    is_Parallel = (abs(recent_lane_dydx_p1 - newlane_dydx_p1) < gradient_threshold) & (abs(recent_lane_dydx_p2 - newlane_dydx_p2) < gradient_threshold) & (abs(recent_lane_dydx_p3 - newlane_dydx_p3) < gradient_threshold)

    # Check curvature

    curvature_threshold = 500
    is_SameCurvature = abs(recent_lane.radius_of_curvature - newlane.radius_of_curvature) < curvature_threshold

    if is_Parallel & is_SameCurvature:
        return True
    else:
        return False

### This function computes the mean x-values of the most recent N lane lines

def cal_mean_lane(lanes, N):

    mean_lane = Lane()
    cum_lane_xpts = []

    # build a list of the most recent N lane lines
    for lane in lanes[-min(N, len(lanes)):]:
        cum_lane_xpts.append(lane.allx)

    # compute the mean x-values and the v-values of the lane line
    mean_lane.allx = np.mean(cum_lane_xpts, axis=0)
    mean_lane.ally = lanes[-1].ally

    # compute the other attributes of the mean lane line
    lanefit = np.polyfit(mean_lane.ally, mean_lane.allx, 2)
    mean_lane.polycof = lanefit
    mean_lane.radius_of_curvature = cal_curvature(lanefit)

    return mean_lane

### This function is the processing pipeline for the lane line detection algorithm

def process_image(image):

    global left_lanes, right_lanes, bad_left_lanes, bad_right_lanes
    global left_failed_count, right_failed_count

    # Step 1: correct the distortion on the raw images using the camera calibration results
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    # Step 2: create a thresholded binary image of the raw color image using color transform and
    # gradient thresholding
    result = pipeline(undistorted)
    # Step 3: perform a perspective transformation on the binary image
    binary_warped, perspective_M, Minv = warp(result)
    # Step 4: detect the left and right lane lines on the warped binary image
    left_lane, right_lane, out_img = detect_lanes(binary_warped)
    # Step 5: perform sanity check on the newly detected lane lines using historical data
    # Left lane first
    if (len(left_lanes) > 0):
        left_is_fit = sanity_check(left_lanes[-1], left_lane)
        if left_is_fit == True:  # Sanity check passes
            if len(bad_left_lanes) > 0:
                bad_left_lanes.clear()
            left_failed_count = 0
            left_lanes.append(left_lane)
            if len(left_lanes) > 10:
                left_lanes.pop(0)
        else:  # Sanity check fails
            left_failed_count += 1
            bad_left_lanes.append(left_lane)
            #  If sanity check fails more than 5 times in a row, reset the lane list with the mean of the last
            #  2 lanes
            if left_failed_count > 5:
                newlane = cal_mean_lane(bad_left_lanes, 2)
                bad_left_lanes.clear()
                left_lanes.clear()
                left_lanes.append(newlane)
                left_failed_count = 0
    else:
        left_lanes.append(left_lane)

    # Do the same with the right lane
    if (len(right_lanes) > 0):
        right_is_fit = sanity_check(right_lanes[-1], right_lane)
        if right_is_fit == True:  # Sanity check passes
            if len(bad_right_lanes) > 0:
                bad_right_lanes.clear()
            right_failed_count = 0
            right_lanes.append(right_lane)
            if len(right_lanes) > 10:
                right_lanes.pop(0)
        else:  # Sanity check fails
            right_failed_count += 1
            bad_right_lanes.append(right_lane)
            #  If sanity check fails more than 5 times in a row, reset the lane list with the mean of the last
            #  2 lanes
            if right_failed_count > 5:
                newlane = cal_mean_lane(bad_right_lanes, 2)
                bad_right_lanes.clear()
                right_lanes.clear()
                right_lanes.append(newlane)
                right_failed_count = 0
    else:
        right_lanes.append(right_lane)

    # Step 6: draw lane lines on a blank image
    color_warp, final_left_lane, final_right_lane = draw_lanes(binary_warped)
    # Step 7: warp the blank image back to the original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0]))
    # Step 8: combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    # Step 9: calculate deviation from center
    midpoint = int((final_left_lane.allx[-1] + final_right_lane.allx[-1]) / 2)
    car_center = int(color_warp.shape[1] / 2)
    midpoint_deviation_pix = abs(car_center - midpoint)
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    midpoint_deviation_m = midpoint_deviation_pix * xm_per_pix
    # Step 10： display the curvature and deviation results
    cv2.putText(result, "Radius of Curvature (left lane): " + str(int(final_left_lane.radius_of_curvature)) + "m", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.putText(result, "Radius of Curvature (right lane): " + str(int(final_right_lane.radius_of_curvature)) + "m", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.putText(result, "Deviation from center: " + str(float("{0:.2f}".format(midpoint_deviation_m))) + "m", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    return result

left_lanes, bad_left_lanes = [], []
right_lanes, bad_right_lanes = [], []
left_failed_count = 0
right_failed_count = 0

# Use code below for processing images
# result = process_image(img)
# plt.imshow(result)
# plt.show()

# Use code below for rendering videos
output = 'output_images/video_2_test.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)