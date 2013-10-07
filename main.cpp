#include <cv.h>
#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include<math.h>
#include "Utils.h"

using namespace cv;
using namespace std;

vector<Mat> listImages(char *ptr);
void global_threshold(vector<Mat>& images);
Mat histogramA(vector<Mat>& A,int n);
Mat histogramA(Mat& A);

void drawhistogram(Mat& A);
void applyGaussian(cv::Mat &input, cv::Mat &output);
void findOptimumThreshold(int index,Mat &img,Mat& histData);
void slidingWindow(Mat& img, int step);
int main( int argc, char** argv )
{

  if( argc != 2 )
   {
      printf( "No image data %s\n",argv[1] );
      return -1;
  }
  //getting the size of the image,
  vector<Mat>images = listImages(argv[1]);

  printf("images size: %d\n",images.size());
  int l = images.size();
  int n = images[0].rows;
  int r = images[0].cols;
  int s = ceil(log2(n));
  Mat Var = Mat::zeros(s,7, CV_32F);//V stores the different types of variances for the different window sizes used afterwards to find the optimum window size


  //implemenation of global method
  float window = n;
  Var.at<float>(s-1,0)=window;

  //cout<<"Var ="<<Var<<endl;
  cout<<"HEYE: "<<pow((double)2,(double)5)<<endl;
  global_threshold(images);
  for(int j=0;j<images.size();j++){
    for(int i=1;pow((double)2,i)<=n;i=i++){
      cout<<"window="<<pow((double)2,i)<<" for image="<<j<<endl;
      slidingWindow(images[j],4);
    }
  }
  return 0;
}
/**
 * Used to avoid noise in the image.
 */
void applyGaussian(cv::Mat &input, cv::Mat &output) {
    double sigma = 1.5;
    cv::Mat gaussKernel = cv::getGaussianKernel(9,sigma,CV_32F);
    cv::GaussianBlur( input, output, cv::Size(3,3), 1.5);
}
void score_images(vector<Mat>& images){
  int width = 16;
   int height = images.size()/width;
   Mat histA = Mat::zeros(256,1, CV_32F);
   Mat A = Mat::zeros(256,1, CV_32F);
   //calculate the histogram for the entire image
   for(int j=0;j<height;j++){
       std::vector<Mat>   subIm(&images[j*width],&images[(j*width+width)-1]);
       Mat h=histogramA(subIm,width);
       Mat out;
       add(histA,h,out);
       histA = out;
   }
   cout<<"output="<<histA<<endl;

    double mean= cv::mean( histA ).val[0];
    cout<<"mean = "<<mean<<endl;
    float sum = 0;
    for(int i=0;i<histA.rows;i++){
        cout<<"p="<<*histA.ptr<float>(0,i)<<endl;
        sum += pow((*histA.ptr<float>(0,i)-mean),2);
    }
    float var = sqrt(sum/histA.cols);
    cout<<"var = "<<var<<endl;

    drawhistogram(histA);
}
void global_threshold(vector<Mat>& images){
    int width = 16;
     int height = images.size()/width;
     Mat histA = Mat::zeros(256,1, CV_32F);
     //calculate the histogram for the entire image
     for(int j=0;j<height;j++){
         std::vector<Mat>   subIm(&images[j*width],&images[(j*width+width)-1]);
         Mat h=histogramA(subIm,width);
         Mat out;
         add(histA,h,out);
         histA = out;
     }


     for(int i = 0;i<images.size();i++)
       findOptimumThreshold(i,images[i],histA);

}

Mat histogramA(vector<Mat>& A,int n){
  Mat *a = &A[0];


  Mat out;
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  int histSize = 256;
  int channels = 1;
  calcHist( a, n, &channels, Mat(),out,1,&histSize, &histRange,true, true );
  return out;
}
Mat histogramA(Mat& A){
  Mat *a = &A;
  int n =1;


  Mat out;
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  int histSize = 256;
  int channels = {0};
  calcHist( &A, 1, &channels, Mat(),out,1,&histSize, &histRange,true, true );
  return out;
}
void drawhistogram(Mat& A){
  int histSize = 256;

  // Draw the histograms image array
   int hist_w = 512; int hist_h = 400;
   int bin_w = cvRound( (double) hist_w/histSize );

   Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

   //normalize to have values fall in windows
   normalize(A, A, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

   for( int i = 1; i < histSize; i++ )
   {
       line( histImage, Point( bin_w*(i-1), hist_h - cvRound(A.at<float>(i-1)) ) ,
                             Point( bin_w*(i), hist_h - cvRound(A.at<float>(i)) ),
                             Scalar( 255, 0, 0), 2, 8, 0  );
   }
   /// Display
  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
  imshow("calcHist Demo", histImage );
  waitKey(0);
}

vector<Mat> listImages(char *ptr)
{
    DIR *dp;
    struct   dirent *d;
    char full_path[200];
    vector<Mat> images;

    //printf("checkpoint 1\n");
    if((dp = opendir(ptr)) == NULL){
            //printf("Error %s\n",ptr);
            perror("opendir");
            return images;
    }

    while((d = readdir(dp)) != NULL)
    {
            //printf("checkpoint 2\n");
            if(!strcmp(d->d_name,".") || !strcmp(d->d_name,".."))
                    continue;

            //Constuct full path
            strcpy(full_path,ptr);
            strcat(full_path,"/");
            strcat(full_path,d->d_name);
            if (d->d_type == DT_DIR)
            {
                    continue;
            }
            else
            {
              Mat gray = imread(full_path,CV_LOAD_IMAGE_GRAYSCALE);
              applyGaussian(gray,gray);
              images.push_back(gray);
            }
    }
    return images;
}

void findOptimumThreshold(int index, Mat &img, Mat &histData){
      int nbHistLevels = 256;


     // total number of pixels
     int total = img.cols*img.rows;

     float sum = 0;
     int t;
     for (t=0; t < nbHistLevels; t++)
        sum += t * *histData.ptr<float>(0,t);

     float sumB = 0;
     int wB = 0;
     int wF = 0;
     float varMax = 0;
     float varMaxOtsu = 0;
     float varMaxEqlVar = 0;
     float varB = 0;
     float varF = 0;
     int thresholdotsu = 0;
     int thresholdequalvar = 10000;
     int threshold = 0;
     for (t=0; t < nbHistLevels; t++) {
        wB += *histData.ptr<float>(0,t);               // Weight Background
        if (wB == 0)
           continue;

        wF = total - wB;                 // Weight Foreground
        if (wF == 0)
           break;

        sumB += (float) (t * *histData.ptr<float>(0,t));

        float mB = sumB / wB;            // Mean Background
        float mF = (sum - sumB) / wF;    // Mean Foreground
        float sB = wF*pow(mB,2)+wF*pow(mF,2);
        float sF = 2*wB*mF*mB;
        // Calculate Between Class Variance
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        // Check if new maximum found
        if (varBetween > varMax) {
           varMaxOtsu = varBetween;
           thresholdotsu = t;
           varB =sB;
           varF =sF;

        }
        if(mB==mF){
            varMaxEqlVar = varBetween;
            thresholdequalvar = t;
            varB =sB;
            varF =sF;
        }
     }
     if(thresholdequalvar<thresholdotsu){
         printf("eq\n");
       threshold = thresholdequalvar;
       varMax = varMaxEqlVar;
     }else{
         threshold = thresholdotsu;
         varMax = varMaxOtsu;
     }
     //fprintf(stdout, "threshold=%d\n", threshold);
     Mat img_al;
     cv::threshold(img, img_al, threshold, 255, CV_THRESH_BINARY);
     /// Display
    //namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    //imshow("calcHist Demo", img_al);
    //waitKey(0);
     char buff[200];
     sprintf(buff, "images/%d.png", index);
     std::string buffAsStdStr = buff;
     //printf(buff);
     imwrite(buff,img_al);

}
void slidingWindow(Mat& img, int step){
  //printf("total size: (%d, %d)",img.cols,img.rows);
  for(int i = 0;i<img.rows;i=i+step){
      for(int j = 0;j<img.cols;j=j+step){
          int w = i+step;
          int h = j+step;
           //printf("1 (%d,%d) to (%d,%d) \n",i,j,w,h);
           cv::Mat subImg = img(cv::Rect(i,j,step,step));

           //printf("2 (%d,%d) to (%d,%d) \n",i,j,w,h);

           img.copyTo(subImg);
           //printf("size=(%d,%d) \n",subImg.cols,subImg.rows);
           //cout<<"subImg "<<subImg<<endl;
           Mat hist = histogramA(subImg);
           findOptimumThreshold(i*j,subImg,hist);
       }
  }
  //cv::Mat subImg = img(cv::Range(0, 0), cv::Range(100, 100));
}
