/*
 * Utils.h
 *
 *  Created on: Oct 4, 2013
 *      Author: radzell
 */
#include <cv.h>
#include <highgui.h>
#ifndef UTILS_H_
#define UTILS_H_

class Utils
{
public:
  Utils();
  static int sumMatrixRow(cv::Mat m,int row,int start, int end);
  static int sumMatrixColumn(cv::Mat m,int col,int start, int end);
  static cv::Mat createUniformRowMat(int start, int end);
  virtual
  ~Utils();
};

#endif /* UTILS_H_ */
