#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv; 
using namespace std; 

cv::Mat cvt_from_srgb_to_linrgb(cv::Mat& srgb_image){

    cv::Mat lin_image(srgb_image.rows, srgb_image.cols, CV_32FC3);
    

    for (int i = 0; i < srgb_image.rows; i++) {
        for (int j = 0; j < srgb_image.cols; j++) {

            cv::Vec3f pixel = srgb_image.at<cv::Vec3b>(i, j);
            pixel/=255.;

            cv::Vec3f linrgb_color;

            for(int k = 0; k < 3; k++){

                double color = pixel[k];

                if(color <= 0.04045){

                    linrgb_color[k] = color / 12.92;
                }
                else{

                    linrgb_color[k] = cv::pow((color + 0.055) / 1.055, 2.4);
                }
            }

            lin_image.at<cv::Vec3f>(i, j) = linrgb_color;
            
        }
    }

    return lin_image;

}

cv::Mat cvt_from_linrgb_to_srgb(cv::Mat& lin_image) {

    cv::Mat srgb_image(lin_image.rows, lin_image.cols, CV_8UC3);

    for (int i = 0; i < lin_image.rows; i++) {
        for (int j = 0; j < lin_image.cols; j++) {

            cv::Vec3f pixel = lin_image.at<cv::Vec3f>(i, j);

            cv::Vec3f srgb_color;

            for(int k = 0; k < 3; k++){

                double color = pixel[k];

                if (color <= 0.0031308) {
                    srgb_color[k] = color * 12.92;
                } else {
                    srgb_color[k] = 1.055 * std::pow(color, 1.0 / 2.4) - 0.055;
                }
            }

            srgb_color*=255;

            srgb_image.at<cv::Vec3b>(i, j) = srgb_color;
            
        }
    }

    return srgb_image;

}

void convertTo8UC3(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat temp;
    src.convertTo(temp, CV_32F, 255.0);
    temp.convertTo(dst, CV_8UC3);
}

double calculateRMSE(const cv::Mat& img1, const cv::Mat& img2) {

    cv::Mat diff, mask;
    cv::absdiff(img1, img2, diff);
    diff = diff.mul(diff);

    cv::Scalar s = cv::sum(diff);
    double sse = s[0] + s[1] + s[2]; 
    cv::cvtColor(img2, mask, cv::COLOR_BGR2GRAY);

    double mse = sse / cv::countNonZero(mask) / 3; 
    return std::sqrt(mse); 

}

void RMSE(cv::Mat pic) {
    vector<cv::Mat> etalon, pic_colors;
    vector<cv::Point> points{cv::Point(128,137), cv::Point(130, 159), cv::Point(130,181), cv::Point(130,204), cv::Point(132,226), cv::Point(132,247)};
    vector<cv::Scalar> colors{cv::Scalar(252,252,252), cv::Scalar(230,230,230), cv::Scalar(200,200,200), cv::Scalar(143,143,143), cv::Scalar(100,100,100), cv::Scalar(50,50,50)};
    for(int i = 0; i < 6; ++i){   
        cv::Mat mask(pic.rows, pic.cols, CV_8UC3, cv::Scalar(0)), tmp(pic.rows, pic.cols, CV_8UC3, colors[i]), pic_color;
        circle(mask, points[i], 5, cv::Scalar(255, 255, 255), -1);
        cv::bitwise_and(tmp, mask, tmp);
        cv::bitwise_and(pic, mask, pic_color);
        etalon.push_back(tmp);
        pic_colors.push_back(pic_color);
    }
    for (int i = 0; i < 6; ++i){
        cout << "RMSE for " << colors[i] << " = " << fixed << setprecision(2) << calculateRMSE(pic_colors[i], etalon[i]) << std::endl;
    }
}

int main(int argc, char** argv) {

    String inputImagePath = "../prj.lab/lab09/input.jpg";
    Mat inputImage = imread(inputImagePath, IMREAD_COLOR);

    Mat lin_picture = cvt_from_srgb_to_linrgb(inputImage);

    Mat tmp;
    convertTo8UC3(lin_picture, tmp);
    imwrite("../prj.lab/lab09/lin_input.jpg", tmp);

    vector<cv::Mat> channels;
    split(lin_picture, channels);
    Scalar mean_scalar = cv::mean(lin_picture);
    double mean = (mean_scalar[0] + mean_scalar[1] + mean_scalar[2]) / 3;

    for (auto& channel : channels) {
        double channel_mean = mean / cv::mean(channel)[0];
        //cout << channel_mean << "\n";
        channel *= channel_mean;
    }
    Mat lin_grayworld;
    merge(channels, lin_grayworld);

    Mat tmp2;
    convertTo8UC3(lin_grayworld, tmp2);
    imwrite("../prj.lab/lab09/grayworld_lin_input.jpg", tmp2);

    Mat grayworld_srgb = cvt_from_linrgb_to_srgb(lin_grayworld);

    imwrite("../prj.lab/lab09/new_pic.jpg", grayworld_srgb);

    vector<Mat> masks;
    vector<Point> points{Point(128,137), Point(130, 159), Point(130,181), Point(130,204), Point(132,226), Point(132,247)};
    for(auto center: points){   
        Mat mask(inputImage.rows, inputImage.cols, CV_8UC3, Scalar(0));
        circle(mask, center, 5, Scalar(255, 255, 255), -1);
        masks.push_back(mask);
    }
    RMSE(inputImage);
    Mat input3lab1 = imread("../prj.lab/lab03/output/splited_channel_picture.jpg", IMREAD_COLOR);
    RMSE(input3lab1);
    Mat input3lab2 = imread("../prj.lab/lab03/output/second_way_picture.jpg", IMREAD_COLOR);
    RMSE(input3lab2);

    return 0;
}
