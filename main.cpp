//--SURF code available at opencv tutorial
//--reference http://docs.opencv.org/doc/tutorials/features2d/feature_detection/feature_detection.html //
//--modified to suit our needs
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <conio.h>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/cvaux.h>

using namespace cv;
using namespace std;

void readme();

//-- normalising the points
Mat normalise2Dpts(vector<Point2f> & pts, vector<Point2f> & newpts)
{
 Scalar centroids=mean(pts);
 vector<Point2f> T_pts;
    vector<float> dist(pts.size());
    subtract(pts,centroids,T_pts);

    for (size_t i=0;i<pts.size();i++){
     dist[i]=sqrt(T_pts[i].x*T_pts[i].x+T_pts[i].y*T_pts[i].y);
    }

    Scalar meandist=mean(dist);
    float scale=sqrt(2)/meandist.val[0];

    Mat T(3,3,CV_32FC1);
    cout<<"**************"<<endl;
    T.at<float>(0,0)=scale; T.at<float>(0,1)=0; T.at<float>(0,2)=-scale*centroids.val[0];
    T.at<float>(1,0)=0; T.at<float>(1,1)=scale; T.at<float>(1,2)=-scale*centroids.val[1];
    T.at<float>(2,0)=0; T.at<float>(2,1)=0; T.at<float>(2,2)=1;

    cout<<T.at<float>(0,0)<<'\t'<<T.at<float>(0,1)<<'\t'<< T.at<float>(0,2)<<endl<<
           T.at<float>(1,0)<<'\t'<<T.at<float>(1,1)<<'\t'<< T.at<float>(1,2)<<endl<<
           T.at<float>(2,0)<<'\t'<<T.at<float>(2,1)<<'\t'<< T.at<float>(2,2)<<endl;

    cout<<"**************"<<endl;
    return T;
}

//-- main function
int main( int argc, char** argv )
{
 
//////////////                   SURF alorithm                         ////////////////////////////////////////////
  Mat img_object = imread( "imL.png", CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_scene = imread( "imR.png", CV_LOAD_IMAGE_GRAYSCALE );

  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );
  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  //-- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;
  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );
  drawKeypoints( img_object, keypoints_object, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_scene, keypoints_scene, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
  imshow("Keypoints 2", img_keypoints_2 );
  
  SurfDescriptorExtractor extractor;
  Mat descriptors_object, descriptors_scene;
  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );


  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );
  float max_dist = 0; float min_dist = 100;

  
  for( int i = 0; i < descriptors_object.rows; i++ )
  { float dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }
  /*  //-- Draw only "good" matches
  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Show detected matches
  imshow( "Good Matches", img_matches ); */

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }
 /* // Convert keypoints into Point2f
  std::vector<int> pointIndexesLeft;
    std::vector<int> pointIndexesRight;
    for (std::vector<cv::DMatch>::const_iterator it= matches.begin(); it!= matches.end(); ++it) {

         // Get the indexes of the selected matched keypoints
         pointIndexesLeft.push_back(it->queryIdx);
         pointIndexesRight.push_back(it->trainIdx);
    }
  std::vector<cv::Point2f> selPointsLeft, selPointsRight;
  cv::KeyPoint::convert(keypoints_object,selPointsLeft,pointIndexesLeft);
  cv::KeyPoint::convert(keypoints_scene,selPointsRight,pointIndexesRight);*/

  Mat F=findFundamentalMat(obj, scene,FM_RANSAC,3,0.99);
  cout<<"determinant:"<<determinant(F);

  //rectification of images
  Mat H1;
  Mat H2;
  double threshold;
  stereoRectifyUncalibrated(obj, scene, F, img_object.size(), H1, H2,threshold=5);
  int row=H1.rows;
  int col=H1.cols;
	
  Mat rectified1(img_object.size(), img_object.type());
  warpPerspective(img_object, rectified1, H1, img_object.size());
  imwrite("rectified1.jpg", rectified1);
  
  Mat rectified2(img_scene.size(), img_scene.type());
  warpPerspective(img_scene, rectified2, H2, img_scene.size());
  imwrite("rectified2.jpg", rectified2);
////////////////Loop and Zhang shearing http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=786928 //////////////////////
  Size s1= rectified1.size();
  Size s2= rectified2.size();
  int h1= s1.height;
  int w1= s1.width;
  int h2= s2.height;
  int w2= s2.width;
  int i1=w1-1; int i2=i1/2;
  int j1=h1-1; int j2=j1/2;
  typedef Point_<float> Point3ff;
  Point3ff a1(i2, 0);
  Point3ff b1(i1, j2);
  Point3ff c1(i2, j1);
  Point3ff d1(0, j2);

  Point3ff a2(i2, 0);
  Point3ff b2(i1, j2);
  Point3ff c2(j2, i1);
  Point3ff d2(0, j2);
  
  vector<Point3ff> inPts1, outPts1,inPts2,outPts2;
  inPts1.push_back(a1);
  inPts1.push_back(b1);
  inPts1.push_back(c1);
  inPts1.push_back(d1);
  inPts2.push_back(a2);
  inPts2.push_back(b2);
  inPts2.push_back(c2);
  inPts2.push_back(d2);
  perspectiveTransform(inPts1, outPts1, H1);
  perspectiveTransform(inPts2, outPts2, H2);
 //Point3ff x2=outPts2[1]-outPts2[3];
// Point3ff y2=outPts2[2]-outPts2[0];
 Point3ff x1=outPts1[1]-outPts1[3];
 Point3ff y1=outPts1[2]-outPts1[0];
 //(h²x[1]² + w²y[1]²)/(hw(x[1]y[0] - x[0]y[1]))
 double n1= h1*h1 *x1.y*x1.y; double n2= w1*w1 *y1.y*y1.y;
 double n3= h1*w1*x1.y*y1.x; double n4= h1*w1*x1.x*y1.y;
 double k1= (n1+n2)/(n3-n4);
 k1= abs(k1);
 //(h²x[0]x[1] + w²y[0]y[1])/(hw(x[0]y[1] - x[1]y[0]))
 n1= h1*h1*x1.x*x1.y; n2= w1*w1*y1.x*y1.y;
 n3= h1*w1*x1.x*y1.y; n4= h1*w1*x1.y*y1.x;
 double k2= (n1+n2)/(n3-n4);
 k2= abs(k2);
 Mat S1 = (Mat_<double>(3,3) << k1, k2, 0, 0, 1, 0, 0, 0, 1);
 Mat rectified1_1(img_object.size(), img_object.type());
 warpPerspective(rectified1, rectified1_1, S1, img_object.size());
 imwrite("rectified1_1.jpg", rectified1_1);
  x1=outPts2[1]-outPts2[3];
  y1=outPts2[2]-outPts2[0];
 
  n1= h1*h1 *x1.y*x1.y;  n2= w1*w1 *y1.y*y1.y;
  n3= h1*w1*x1.y*y1.x;  n4= h1*w1*x1.x*y1.y;
  k1= (n1+n2)/(n3-n4);
 k1= abs(k1);
 n1= h1*h1*x1.x*x1.y; n2= w1*w1*y1.x*y1.y;
 n3= h1*w1*x1.x*y1.y; n4= h1*w1*x1.y*y1.x;
 k2= (n1+n2)/(n3-n4);
 k2= abs(k2);
 Mat S2 = (Mat_<double>(3,3) << k1, k2, 0, 0, 1, 0, 0, 0, 1);
 Mat rectified2_1(img_object.size(), img_object.type());
 warpPerspective(rectified2, rectified2_1, S1, img_object.size());
 imwrite("rectified2_1.jpg", rectified2_1);
  Mat img_rectified1 = imread( "rectified1_1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_rectified2 = imread( "rectified2_1.jpg", CV_LOAD_IMAGE_GRAYSCALE );
 // /////////////////////////////////////////////////////////////////////////////////////
  Mat disp,disp8;
  StereoBM sbm;
  sbm.state->SADWindowSize = 9;
        sbm.state->numberOfDisparities =64;
        sbm.state->preFilterSize = 5;
        sbm.state->preFilterCap = 61;
        sbm.state->minDisparity = -26;
        sbm.state->textureThreshold = 507;
        sbm.state->uniquenessRatio = 0;
        sbm.state->speckleWindowSize = 0;
        sbm.state->speckleRange = 8;
        sbm.state->disp12MaxDiff = 1;
        
  sbm(img_rectified1,img_rectified2,disp);
  normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
  imshow("disp",disp8);
  
  waitKey(0);
  return 0;

  }

 
