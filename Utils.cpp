/*
 * Utils.cpp
 *
 *  Created on: Oct 4, 2013
 *      Author: radzell
 */

#include "Utils.h"

Utils::Utils()
{
  // TODO Auto-generated constructor stub

}

Utils::~Utils()
{
  // TODO Auto-generated destructor stub
}

int Utils::sumMatrixRow(cv::Mat m,int row,int start, int end)
{
  int sum =0;
  for(int i =start;i<(end+1);i++){
      sum+= *m.ptr<float>(row,i);
  }
  return sum;
}
int Utils::sumMatrixColumn(cv::Mat m,int col,int start, int end)
{
  int sum =0;
  for(int i =start;i<(end+1);i++){

        sum+= *m.ptr<float>(i,col);
  }
  return sum;
}
cv::Mat Utils::createUniformRowMat(int start, int end){
  end=end+1;
  cv::Mat A = cv::Mat::zeros(end-start,1, CV_32F);
  for(int i=0;i<end-start;i++){
      A.at<float>(i,0) = start+i;
  }

  return A;
}

