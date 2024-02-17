#ifndef CHEEKS_HPP
#define CHEEKS_HPP

int CheeksDetection(std::vector<cv::Rect>& tmpRect, cv::Mat& tmpImg);

int CheeksDetection(std::vector<cv::Rect>& tmpRect, cv::Mat& tmpImg, std::string& tmpmask);

#endif // CHEEKS_HPP