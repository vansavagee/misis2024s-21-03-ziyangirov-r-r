#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv; 
using namespace std; 

std::vector<int> calculateHistogram(const Mat& channel) {
    std::vector<int> histogram(256, 0);

    for (int y = 0; y < channel.rows; ++y) {
        const uchar* row_ptr = channel.ptr<uchar>(y);
        for (int x = 0; x < channel.cols; ++x) {
            uchar pixel_value = row_ptr[x];
            histogram[pixel_value]++;
        }
    }

    return histogram;
}

std::pair<int, int> findNewRange(const std::vector<int>& histogram, double quantile) {
    int low = 0, high = 255;
    const int total_pixels = static_cast<int>(sum(histogram)[0]);
    int threshold = total_pixels * quantile;

    int accumulated_pixels = 0;
    for (int i = 0; i < 256; ++i) {
        accumulated_pixels += histogram[i];
        if (accumulated_pixels >= threshold) {
            low = i;
            break;
        }
    }

    accumulated_pixels = 0;
    for (int i = 255; i >= 0; --i) {
        accumulated_pixels += histogram[i];
        if (accumulated_pixels >= threshold) {
            high = i;
            break;
        }
    }

    return std::make_pair(low, high);
}

Mat applyContrast(const Mat& channel, double quantile) {
    std::vector<int> histogram = calculateHistogram(channel);

    std::pair<int, int> new_range = findNewRange(histogram, quantile);

    Mat result;
    double scale = 255.0 / (new_range.second - new_range.first);
    double offset = -new_range.first * scale;
    channel.convertTo(result, CV_8U, scale, offset);

    return result;
}

Mat3b contrast(const Mat3b& image, double quantile) {
    Mat3b result;
    std::vector<Mat> channels;
    split(image, channels);

    for (int i = 0; i < 3; ++i) {
        Mat processed_channel = applyContrast(channels[i], quantile);
        channels[i] = processed_channel;
    }

    merge(channels, result);
    return result;
}

Mat1b applyContrast_gray(const Mat1b& channel, double quantile) {
    std::vector<int> histogram(256, 0);

    for (int y = 0; y < channel.rows; ++y) {
        const uchar* row_ptr = channel.ptr<uchar>(y);
        for (int x = 0; x < channel.cols; ++x) {
            uchar pixel_value = row_ptr[x];
            histogram[pixel_value]++;
        }
    }
    std::pair<int, int> new_range = findNewRange(histogram, quantile);
    Mat1b result;
    double scale = 255.0 / (new_range.second - new_range.first);
    double offset = -new_range.first * scale;
    channel.convertTo(result, CV_8U, scale, offset);
    return result;
}

void make_output(Mat3b &inputImage, Mat3b &contrasted_image){
    std::vector<Mat> channels;
    split(inputImage, channels);

    Mat1b hists[3];//(256, 256, 230)
    for (int j = 0; j < 3; ++j) {
        hists[j] = Mat1b(256, 256, 230);
        std::vector<int> histogram = calculateHistogram(channels[j]);
        int max_count = *std::max_element(histogram.begin(), histogram.end());

        float scale = static_cast<float>(230) / max_count;
        
        for (int i = 0; i < 256; ++i) {
            Point start_point(i, 256);
            Point end_point(i, 256 - histogram[i] * scale);
            line(hists[j], start_point, end_point, 0, 1);
        }
        if(j != 0){
            hconcat(hists[0], hists[j], hists[0]);
        }
    }
    std::vector<Mat> con_channels;
    split(contrasted_image, con_channels);
    Mat1b con_hists[3];
    for (int j = 0; j < 3; ++j) {
        con_hists[j] = Mat1b(256, 256, 230);
        std::vector<int> histogram = calculateHistogram(con_channels[j]);
        int max_count = *std::max_element(histogram.begin(), histogram.end());

        float scale = static_cast<float>(230) / max_count;
        
        for (int i = 0; i < 256; ++i) {
            Point start_point(i, 256);
            Point end_point(i, 256 - histogram[i] * scale);
            line(con_hists[j], start_point, end_point, 0, 1);
        }
        if(j != 0){
            hconcat(con_hists[0], con_hists[j], con_hists[0]);
        }
    }
    vconcat(hists[0], con_hists[0], hists[0]);
    imwrite("../prj.lab/lab03/output/splited_channel_histogram.jpg", hists[0]);
    imwrite("../prj.lab/lab03/output/splited_channel_picture.jpg", contrasted_image);
}

void make_second_output(Mat3b &inputImage, Mat3b &contrasted_image){
    std::vector<Mat> channels;
    split(inputImage, channels);

    Mat1b hists[3];//(256, 256, 230)
    for (int j = 0; j < 3; ++j) {
        hists[j] = Mat1b(256, 256, 230);
        std::vector<int> histogram = calculateHistogram(channels[j]);
        int max_count = *std::max_element(histogram.begin(), histogram.end());

        float scale = static_cast<float>(230) / max_count;
        
        for (int i = 0; i < 256; ++i) {
            Point start_point(i, 256);
            Point end_point(i, 256 - histogram[i] * scale);
            line(hists[j], start_point, end_point, 0, 1);
        }
        if(j != 0){
            hconcat(hists[0], hists[j], hists[0]);
        }
    }
    std::vector<Mat> con_channels;
    split(contrasted_image, con_channels);
    Mat1b con_hists[3];
    for (int j = 0; j < 3; ++j) {
        con_hists[j] = Mat1b(256, 256, 230);
        std::vector<int> histogram = calculateHistogram(con_channels[j]);
        int max_count = *std::max_element(histogram.begin(), histogram.end());

        float scale = static_cast<float>(230) / max_count;
        
        for (int i = 0; i < 256; ++i) {
            Point start_point(i, 256);
            Point end_point(i, 256 - histogram[i] * scale);
            line(con_hists[j], start_point, end_point, 0, 1);
        }
        if(j != 0){
            hconcat(con_hists[0], con_hists[j], con_hists[0]);
        }
    }
    vconcat(hists[0], con_hists[0], hists[0]);
    imwrite("../prj.lab/lab03/output/second_way_histogram.jpg", hists[0]);
    imwrite("../prj.lab/lab03/output/second_way_picture.jpg", contrasted_image);
}

void make_output_gray(Mat1b &inputImage, Mat1b &contrasted_image){

    Mat1b hists;
    hists = Mat1b(256, 256, 230);
    std::vector<int> histogram = calculateHistogram(inputImage);
    int max_count = *std::max_element(histogram.begin(), histogram.end());

    float scale = static_cast<float>(230) / max_count;
    
    for (int i = 0; i < 256; ++i) {
        Point start_point(i, 256);
        Point end_point(i, 256 - histogram[i] * scale);
        line(hists, start_point, end_point, 0, 1);
    }

    Mat1b con_hists;
    con_hists = Mat1b(256, 256, 230);
    std::vector<int> con_histogram = calculateHistogram(contrasted_image);
    int con_max_count = *std::max_element(con_histogram.begin(), con_histogram.end());

    float con_scale = static_cast<float>(230) / con_max_count;
    
    for (int i = 0; i < 256; ++i) {
        Point start_point(i, 256);
        Point end_point(i, 256 - con_histogram[i] * con_scale);
        line(con_hists, start_point, end_point, 0, 1);
    }

    vconcat(hists, con_hists, hists);
    imwrite("../prj.lab/lab03/output/gray_pic_cont.jpg", contrasted_image);
    imwrite("../prj.lab/lab03/output/gray_histogram.jpg", hists);

}

bool isGrayScale(const Mat& img) {
    if (img.channels() == 1) return true;

    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            Vec3b color = img.at<Vec3b>(r, c);
            if (!((color[0] == color[1]) && (color[1] == color[2]))) {
                return false;
            }
        }
    }
    return true;
}

Mat second_applyContrast(const Mat& channel, int first, int second) {
    Mat result;
    double scale = 255.0 / (second - first);
    double offset = -first * scale;
    channel.convertTo(result, CV_8U, scale, offset);

    return result;
}

Mat3b second_contrast(const Mat3b& image, double quantile) {
    Mat3b result;
    std::vector<Mat> channels;
    split(image, channels);

    int min = 0, max = 255;

    for (int i = 0; i < 3; ++i) {
        std::vector<int> histogram = calculateHistogram(channels[i]);
        std::pair<int, int> new_range = findNewRange(histogram, quantile);
        if(new_range.first > min) min = new_range.first;
        if(new_range.second < max) max = new_range.second;
    }
    for(int i = 0; i<3;++i){
        Mat processed_channel = second_applyContrast(channels[i], min, max);
        channels[i] = processed_channel;
    }
    merge(channels, result);
    return result;
}

int main(int argc, char** argv) {
    
    CommandLineParser parser(argc, argv,
        "{@inputImagePath    | ../prj.lab/lab03/input/color_pic.jpg     | input image path}"
        "{quantile           | 0.15    | quantile}"
    );
    
    String inputImagePath = parser.get<String>("@inputImagePath");
    double quantile = parser.get<double>("quantile");

    Mat3b inputImage = imread(inputImagePath, IMREAD_UNCHANGED);
    Mat3b contrasted_image = contrast(inputImage, quantile);
    Mat3b second_image = second_contrast(inputImage, quantile);
    make_output(inputImage, contrasted_image);
    make_second_output(inputImage, second_image);

    Mat1b gray_input = imread(inputImagePath, IMREAD_GRAYSCALE);
    imwrite("../prj.lab/lab03/input/gray_pic.jpg", gray_input);
    if(isGrayScale(gray_input)) {
        Mat1b gray_output = applyContrast_gray(gray_input, quantile);
        make_output_gray(gray_input, gray_output);
    }
    return 0;
}