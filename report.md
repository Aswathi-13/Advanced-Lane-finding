## Report
---

**Advanced Lane Finding Project**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test6/calibration.jpg "calibration image and undistorted image"
[image2]: ./output_images/test3/original_and_undistorted.jpg "original and undistorted images"
[image3]: ./output_images/test3/binary.jpg "Binary Image"
[image4]: ./output_images/test3/birdseye.jpg "Perspective Transform"
[image5]: ./output_images/test3/img_Polynomial.jpg "Fit Polynomial"
[image6]: ./output_images/test3/final.jpg "Final output image"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here each rubric points are individually considered and described the approach used to implement it

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "p1.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the camera is calibrated, the input image (`test image`) is passed to a function `undistortion` (described under `2. UNDISTORTING THE IMAGE` in the code cell) along with the camera matrix (`mtx`) and distrotion factors(`dist`). When undistortion is applied to a testimage:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at code cell `3. COLOR AND GRADIENT THRESHOLDING`). As the emphasis edges were close to vertical, sobel along x axis `sobelx` is preferred. Also, only the pixel values with certain magnitude and particular orientation was chosen by satisfying the conditions in `magnitude of gradient` and `direction of gradient`. It was also observed that for colour thresholding (HLS), measurement of colourfulness (`S_channel`) did a better job for lane detection than H or L. Here's an example of my output with pixel values satisfying all thresholds. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective Transform defined in the function `persTransform` calls another function `getPerspectiveTransform()` to get the transformation matrix. It takes the source (`src`) and destination (`dst`) points as the inputs. The Python scripts to the same can be found under `4. PERSPECTIVE TRANSFORM` code cell. I chose the hardcode the source and destination points in the following manner:
```python
src = np.float32([[190,720],[600,450],[690,450],[1100,700]])
dest = np.float32([[offset, img_size[1]],[offset, 0],[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]]])
```
with offset value being `300`.
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 300, 720      | 
| 600, 450      | 300,   0      |
| 690, 450      | 980,   0      |
| 1100, 700     | 980, 720      |



The function `getPerspectiveTransform()` can also be used to the inverse matrix (`invM`) to transform back the birds eye view image to real image. Order of inputs changes to (`dst`) and (`src`). With the `image` and transformation matrix `M` as inputs, `wrapPerspective` returned a binary wraped image.
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To locate lines, binary warped image is passed to the function `find_lane_pixel` (defined under cell`5. FINDING THE LANE BOUNDARIES`). To decide which pixel belongs to the left or right lane, a histogram of the lower half of the image is plotted. This emphasis the areas where the binary activations occur across the image. With respect to the mid-point, two peak points are observed which indicates the starting point of the left and right lanes in the *x-axis* respectively. From that point a sliding window is placed around the line centers to find and follow the line up to the top of the frame. The steps are briefed as follows:
* Get the two highest peaks as the starting points of lane lines by using histogram
```python
histogram = np.sum(img[img.shape[0]//2:,:],axis =0)
```
* Find the mid point of the histogram, the highest peak to the left of midpoint = starting point of left lane and to the right is the starting point of the right lane
```python
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:])+midpoint
```
* Defining the required hyper parameters ( from line 13 to 27 under `5. FINDING THE LANE BOUNDARIES`)
* Introduce the sliding window by iterating through `nwindows` to track the curvature
  * Find the coordinates of the sliding window for both left and right lanes
  ```python
        win_y_low = img.shape[0]-(window+1)*window_height
        win_y_high = img.shape[0]-(window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current +margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current +margin
  ```
  * Find the pixel which falls into the activation window and append those pixel indices
  * If the number of pixels are more, then recenter the window
  * When all the indices are tracked, a polynomial is plotted.
   To plot a polynomial, a function `fit_poly` is defined as shown below:
   ```python
    #generating the xy values for plotting
    ploty = np.linspace(0,img.shape[0]-1,img.shape[0])
    out_img = np.dstack((img,img,img))*255
    #gernerating A,B,C coefficients of polynomial using polyfit function
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    
    try:
        #polynomial
        left_fitx = left_fit[0]*ploty**2+ left_fit[1]*ploty +left_fit[2]
        right_fitx = right_fit[0]*ploty**2+ right_fit[1]*ploty +right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
   ```
* To speed up the computational time, it is preferred to find the lane from the prior values once few initial pixel positions are known (function `search_around_poly`). To do that, two global variables (`left_fit_before, right_fit_before`) are declared as zero or null. They acts like a flag. The flag value is checked each time to choose the method to find the lane line. If the values are zero, then sliding window approach is used as an intense search is required. If the flag values are not zero, then mean value of the index position is passed to function `search_around_poly` to find the lane pixels. In this function, the pixel values within a preferable margin around the previous polynomial is checked as shown below:
```python
#finding the activation pixels within +/- margin of the polynomial line
    left_lane_inds = ((nonzerox > (left_fit_before[0]*(nonzeroy**2) + left_fit_before[1]*nonzeroy + left_fit_before[2] - margin)) & (nonzerox < (left_fit_before[0]*(nonzeroy**2) + left_fit_before[1]*nonzeroy + left_fit_before[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit_before[0]*(nonzeroy**2) + right_fit_before[1]*nonzeroy + right_fit_before[2] - margin)) & (nonzerox < (right_fit_before[0]*(nonzeroy**2) + right_fit_before[1]*nonzeroy + right_fit_before[2] + margin)))
```
History of the left and right lane indexes are stacked to global variables. 
        
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is defined under `measure_curvature_real` (under the cell `#6. RADIUS OF CURVATURE` ). Pixel values are then converted from pixel space to real world space by multiplying with 'meters per pixel' ratio. Code snippet is shown below:
```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
# Define y-value where we want radius of curvature
y_eval = np.max(ploty) * ym_per_pix
    
# Implementing the calculation of R_curve (radius of curvature) #####
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

```
Position of the vehicle is mathematically calculated by calling the function `vehicle_position` by passing the image and index values. This is defined in the cell under `7. POSITION OF THE VEHICLE IN METERS`. Code snippet:
```python3
#finding the midpoint
midpt = (x_lt+x_rt)//2
#position of the vehicle
veh_pos = ((undist_img.shape[1]//2) - midpt) * xm_per_pix 
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell `8. WARP BACK TO REAL IMAGE`. Inverse transformation matrix is used to transfer back to real image and radius of curvature and position of the vehicle is annoted in the image as shown in the figure below
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Binary thresholding with combined threshold values took some time to get a decent threshold values. Pipeline didnot work for lanes with intense curves like in harder challenge and lanes like in challenge video.  
For challenge video I guess, it needs more refined binary thresholding to detect the lane lines 
for harder challenge, i guess it is because of the manual selection of source points and destination points. The source points and the destination points are to be detected automatically based on the image size.
