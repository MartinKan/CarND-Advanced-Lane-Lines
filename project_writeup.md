## Udacity Self Driving Car Nanodegree
## Project Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image2]: ./camera_cal/test_undist.jpg "Undistorted"
[image3]: ./test_images/test1.jpg "Road Before Transformed"
[image4]: ./output_images/test1_transformed.jpg "Road After Transformed"
[image5]: ./output_images/test1_binary.jpg "Road After Binary"
[image6]: ./output_images/straight_lines_undistorted.jpg "Road Before Warping"
[image6A]: ./output_images/straight_lines_warped.jpg "Road After Warping"
[image7]: ./output_images/window-topdownonly.jpg "Top Down Windows"
[image8]: ./output_images/window-topandbottom.jpg "Top and Bottom Windows"
[image9]: ./output_images/window-topandbottom-falsepositive.jpg "False positive in sliding windows"
[image10]: ./output_images/test1-final.jpg "Road Final"
[video1]: ./output_images/final_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 28 to 101 of the file "detect.py".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Before calibration
![alt text][image2]

After calibration
![alt text][image2]

The camera calibration and distortion coefficients computed above are then saved to a pickle file (wide_dist_pickle.p) for later use.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]

I first retrieved the camera calibration and distortion coefficients from the pickle file mentioned in the step above and then applied the distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (150 to 255 pixel values of the S channel of the HLS color space) and gradient thresholds (30 to 100 pixel values of the scaled sobel in the X direction) to generate a binary image (thresholding steps at lines 107 through 136 in `detect.py`).  Here's an example of my output for this step:

Before transformation:
![alt text][image4]

After transformation
![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 140 through 163 in the file `detect.py`.  The `warp()` function takes as inputs an image (`img`).  I chose to hardcode the source and destination points in the following manner:

```python

	left_bottom = [188, imshape[0]]	
    left_top = [590, 450]
    right_top = [690, 450]
    right_bottom = [1120, imshape[0]]

    src = np.float32(
        [left_bottom,
         left_top,
         right_top,
         right_bottom])

    dst = np.float32(
         [[200, imshape[0]], 
         [200, 0], 
         [1000, 0], 
         [1000, imshape[0]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 188, 720      | 200, 720        | 
| 590, 450      | 200, 0      |
| 690, 450     | 1000, 0      |
| 1120, 720      | 1000, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Before Transformation:
![alt text][image6]

After Transformation:
![alt text][image6A]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Below are the steps I have taken to identify the lane line pixels and fit their positions with a polynomial:

1. Take a histogram of both the top half and bottom one-ninth (which represents the height of a window) of the image (lines 175 through 179)
2. Find the peak of the left and right halves of the histogram on both the top and bottom of the image to identify the starting point (lines 182 through 191)
3. Identify the non zero pixels in the image and configure the sizes and vertices of the sliding windows (lines 193 through 223)
4. I used a modified version of the sliding window technique to identify the lane lines.  The principle of the sliding window technique is to identify all of the non zero pixel values in a series of vertically adjacent boxes (that we refer to as windows) that move vertically upwards from the bottom to the top of the image.  Once the starting location of the first window is determined, the next window will be placed directly above it with the x values of its vertices sliding left or right (hence the name of the technique) depending on where the mean of the non zero pixel locations fall in the previous window.  But instead of moving vertically in one direction only (bottom up), I had my sliding windows move in two directions: top down and bottom up.  My function would first slide the windows down from the top of the image, halting when I hit the center of the image, before going back up from the bottom to the center. If the windows formed from going top down and bottom up join up in the center of the image, I know that it is very likely for a lane line to be found. Otherwise, if the windows do not join
up, then the top down windows will be wiped and the bottom up windows will continue to iterate until it reaches the top.  I found that by moving in both directions, the windows are able to capture curved lines more accurately, but sometimes it will lead to false positives (e.g. other lines are detected at the top edges of the image) so it is important to check if the windows join up in the center of the image.  (lines 225 through 324)
5. Extract the left and right lane pixel positions identified by the sliding windows and fit a second order polynomial to each of the left and right lane lines (lines 326 through 338)

Bottom Up only Approach
![alt text][image7]

Compare with Combined Top Down and Bottom Up Approach
![alt text][image8]

False Positives in Combined approach
![alt text][image9]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 378 through 394 and lines 551 through 556 in my code in `detect.py`.  For the curvature of the lane lines, I used the scaled derivatives of the lane lines with respect to the maximum y value (i.e. the bottom of the image) and plugged them into the given formula to calculate the radius of the curvature.  For the position of the vehicle with respect to center, I assumed that the camera is placed at the center of the vehicle and then used the midpoint of the lane lines to calculate the vehicle's deviation from the center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 496 through 550 in my code in `detect.py` in the function `process_image()`.  Here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/final_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Apart from the issue I identified above in the section on the sliding window, I also encountered other issues during the implementation of the project.  Some of those issues relate to the adverse effects on the lane detection algorithm during instances where there are high level of noises in the image (caused by shadows) and when the level of contrast between the lane lines and the color of the road surface was not high.  

In the first case, the extra background noises in the image would affect the curvature of the detected lane line, as unwanted pixels were identified as part of the lane lines that would cause the polynomial fitting to be off.  To counter this, I placed a ceiling on the number of pixels in a window（line 212) - in order for the pixels in a window to be counted towards the fitting of the polynomial, the number of pixels in that window has to be lower than the ceiling.  This seemed to have a positive impact on the detection algorithm.

In the second case, the low contrast level between the lane lines and the color of the road surface would sometimes cause part of the lane lines to appear invisible to the detection algorithm.  This would also throw off the polynomial fitting and would cause the lane line extrapolation results to fluctuate wildly from frame to frame.  To counter this, I implemented a few steps: (A) I used a smoothing function that takes the averages of the pixel values from the previous N frames before drawing them back to the screen (lines 460 through 478 of 'detect.py'); (B) I perform a sanity check on new lane lines against previous lane lines to see if they are roughly parallel and have roughly the same curvature (lines 428 to 456); (C) I experimented with different threshold values for the color and gradient transform to find the optimal configuration in low contrast scenarios (line 107).

I experimented my code with the challenging video and it failed to properly identify the lane lines.  My pipeline seems to fail where there are more than one set of lines in an image.  To make my algorithm more robust, I should also include some logic to determine which set of lines in an image / video are most likely to be lane lines and which set of lines are most likely to be false positives.
