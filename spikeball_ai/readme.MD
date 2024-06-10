# SpikeBall AI

Ball detection, spike detection
Then application on video for detection of moments of interest if image-level detectors are satisfactory

Results sheet
https://docs.google.com/spreadsheets/d/11eNwIu8-KOcoUV1y7D9ItPUWB6DJLhUxwq7t-Ndstj8/edit?usp=sharing 

# questions: 
- screen resolution? all frames /2 or /4? first frames full resolution then reduce as we know where the ball is?
- dont search the full screen for the ball once we know where it is for next frames?
- when to work in colors, when to work in colors + pixels 2D, when to work in colors + estimated 3D coords
- estimate depth coord with just the growth/shrink of the ball?
- what to do with frames where ball is invisible?
- time frame resolution, every single frame is redundant, start with every x(3-4?) frame then re-do those in between that are fucked up?
- two label images, one for color values, one for accurate position of ball in 3D, then actually test out the physics modifications+other post process?
- work in [pixel row %, pixel col %, L, a, b]? for better patterns?
- count grey pixels as class negatives or don't count them at all?
- a few regressors or a classifier? mainly ball regressor then remove preds from other regressors?


# parts: 
- take video into frames
- apply pre process (standardize color Lab, something with contrast, sharpness?)
- shift into best data form (color spaces)
- predict ball based on color (sample a wide grid first, then only detail areas around high proba signals)
- modify prediction based on ball's likely position (went straight-ish with same speed, with same linear acceleration, has bounced any time between the two frames and got reflected, add gravity to the acceleration model by just lowering the line)
- modify prediction based on circular-shape (increase probability and not hard cut, for partial obscursions)
- output the ball's center position and estimated radius on each frame, debug output is the prediction frames and its estimated trajectory and re-drawn curve



# steps: 
- get data
- label data
- script to turn labelled images into color dataset

- script to extract normalization info from data (Lab mean+std, )
- script to convert to all color spaces

- script to build and fit all models (just raw first, then adjust hyperparams after selection of reproc and color space)
(- script to stratify based on Lab (merge 3 column stratifications?)?)
- script to take a dataset and test out all combinations of pre-process + color spaces + models

- script to take a video into frames
- script to apply preproc
- script to apply model

- 