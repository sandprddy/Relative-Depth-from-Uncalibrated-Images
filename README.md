# Relative-Depth-from-Uncalibrated-Images

Used SURF feature detector to find correspondence.

Used RANSAC algorithm to calculate fundamental matrix.

Calculated Fundamental matrix is used in Richard Hartley's rectification algorithm to rectify the images.

Richard Hartley's rectification results in shearing of images. This images are rectified using another matrix S (a solution by Loop and Zhang1 effect on image.

Modified H. Hirschmuller algorithm implementation in Opencv(stereoSGBM) is used to calculate disparity map.
