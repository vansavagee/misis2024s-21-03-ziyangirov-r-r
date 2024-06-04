#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>

using namespace cv;
using namespace std;

Mat segmentImage(const Mat& src) {
    Mat colorImage;
    cvtColor(src, colorImage, COLOR_GRAY2BGR);

    Mat blurredImage;
    GaussianBlur(colorImage, blurredImage, Size(33, 33), 0);

    Mat meanShiftedImage;
    pyrMeanShiftFiltering(blurredImage, meanShiftedImage, 20, 40);

    Mat resultImage;
    cvtColor(meanShiftedImage, resultImage, COLOR_BGR2GRAY);
    return resultImage;
}

double estimateSegmentation(const Mat& gt, const Mat& seg, const int& numCircles) {
    double ans = 0;

    for (int i = 0; i < numCircles; ++i) {
        for (int j = 0; j < numCircles; ++j) {
            Mat based(numCircles * 50, numCircles * 50, CV_8UC1, Scalar(0));
            rectangle(based, Point(i * 50, j * 50), Point(i * 50 + 50, j * 50 + 50), Scalar(255), FILLED);

            Mat part_seg, part_gt;
            seg.copyTo(part_seg, based);
            gt.copyTo(part_gt, based);

            Mat intersection;
            bitwise_and(part_seg, part_gt, intersection);
            double intersectionCount = countNonZero(intersection);

            Mat union_;
            bitwise_or(part_seg, part_gt, union_);
            double unionCount = countNonZero(union_);

            double score = (unionCount == 0) ? 0 : intersectionCount / unionCount;
            ans += score;
            
        }
    }

    return ans / (numCircles * numCircles);
}

int main(int argc, char** argv) {
        Mat image = imread("../prj.lab/lab07/input/input3.png", IMREAD_GRAYSCALE);
        Mat gt = imread("../prj.lab/lab07/input/GT3.png", IMREAD_GRAYSCALE);

        Mat binLab4;
        Mat bluredImage;
        cv::GaussianBlur(image, bluredImage, cv::Size(33, 33), 0);

        cv::adaptiveThreshold(bluredImage, binLab4, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 33, 0);

        Mat resultImage = segmentImage(image);


        Mat binGT;
        threshold(gt, binGT, 100, 255, THRESH_BINARY);
        
        threshold(resultImage, resultImage, 100, 255, THRESH_BINARY);

        double mark = estimateSegmentation(binGT, resultImage, 5);
        double mark2 = estimateSegmentation(binGT, binLab4, 5);
        cout << "Result estimation: " << std::fixed << std::setprecision(2) << mark << endl;
        cout << "Result estimation: " << std::fixed << std::setprecision(2) << mark2 << endl;
        imwrite("../prj.lab/lab07/output/segResult3.png", resultImage);
        imwrite("../prj.lab/lab07/output/bin3.png", binLab4);
    return 0;
}