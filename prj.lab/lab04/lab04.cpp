#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#define M_PI 3.14159265358979323846

using namespace cv;
using namespace std;

struct Circle {
    double x, y, r;
};

int minRadius = 8;
int maxRadius = 20;
int numCircles = 7;
int minContrast = 60;
int maxContrast = 255;
int stddev = 25;
int ksize = stddev % 2 == 1 ? stddev + 18 : stddev + 19;
int threshold_ = 35;
std::vector<std::vector<int>> x_y_r;
std::vector<std::vector<int>> x_y_r_detection;

Mat generateSquareClear(int radius) {
    Mat square(50, 50, CV_8UC1, Scalar(0));
    circle(square, Point(25, 25), radius, Scalar(255), -1);

    square.setTo(255, square > 255);
    square.setTo(0, square < 0);

    return square;
}

Mat generateImageClear() {
    vector<Mat> squares;
    for (int i = 0; i < numCircles; ++i) {
        for (int j = 0; j < numCircles; ++j) {
            int radius = minRadius + (maxRadius - minRadius) * (j / (float)(numCircles - 1));
            squares.push_back(generateSquareClear(radius));
        }
    }
    Mat based(numCircles * 50, numCircles * 50, CV_8UC1, Scalar(0));
    for (int i = 0; i < numCircles; ++i) {
        for (int j = 0; j < numCircles; ++j) {
            Mat squareROI = based(Rect(j * 50, i * 50, 50, 50));
            squares[i * numCircles + j].copyTo(squareROI);
        }
    }
    imwrite("../prj.lab/lab04/input/based.png", based);
    return based;
}

Mat generateSquare(int radius, int contrast, int blurAmount) {
    Mat square(50, 50, CV_8UC1, Scalar(80));
    circle(square, Point(25, 25), radius, Scalar(80 + contrast * 175 / 255), -1);
    GaussianBlur(square, square, Size(blurAmount, blurAmount), 0);
    square.setTo(255, square > 255);
    square.setTo(0, square < 0);

    return square;
}

Mat generateImage() {
    vector<Mat> squares;

    for (int i = 0; i < numCircles; ++i) {
        int contrast = minContrast + (maxContrast - minContrast) * (i / (float)(numCircles - 1));

        for (int j = 0; j < numCircles; ++j) {
            int radius = minRadius + (maxRadius - minRadius) * (j / (float)(numCircles - 1));
            x_y_r.push_back({25 + j * 50, 25 + i * 50, radius, 80 + contrast * 175 / 255});
            squares.push_back(generateSquare(radius, contrast, 15));
        }
    }

    Mat image(numCircles * 50, numCircles * 50, CV_8UC1, Scalar(0));
    Mat noise = Mat_<int>(numCircles * 50, numCircles * 50);
    for (int i = 0; i < numCircles; ++i) {
        for (int j = 0; j < numCircles; ++j) {
            Mat squareROI = image(Rect(j * 50, i * 50, 50, 50));
            squares[i * numCircles + j].copyTo(squareROI);
        }
    }
    cv::randn(noise, 0, stddev);
    image += noise;

    imwrite("../prj.lab/lab04/input/image.png", image);

    FileStorage fs("../prj.lab/lab04/input/ground_truth.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    fs << "data" << "{";
    fs << "objects" << "[";

    for (auto elem : x_y_r) {
        fs << "{";
        fs << "p" << "[";
        fs << elem[0] << elem[1] << elem[2];
        fs << "]";
        fs << "c" << elem[3];
        fs << "}";

    }
    fs << "]";
    fs << "background" << "{";
    fs << "size" << "[";
    fs << numCircles * 50 << numCircles * 50;
    fs << "]";
    fs << "color" << 80;
    fs << "blur" << 15;
    fs << "noise" << stddev;
    fs.release();

    return image;
}

double circleIntersectionArea(Circle c1, Circle c2) {
    double d = sqrt((c1.x - c2.x) * (c1.x - c2.x) + (c1.y - c2.y) * (c1.y - c2.y));
    if (d >= c1.r + c2.r) {
        return 0.0; 
    } else if (d <= abs(c1.r - c2.r)) {
        double r_min = min(c1.r, c2.r);
        return M_PI * r_min * r_min; 
    } else {
        double a1 = c1.r * c1.r * acos((d * d + c1.r * c1.r - c2.r * c2.r) / (2 * d * c1.r));
        double a2 = c2.r * c2.r * acos((d * d + c2.r * c2.r - c1.r * c1.r) / (2 * d * c2.r));
        double a3 = 0.5 * sqrt((-d + c1.r + c2.r) * (d + c1.r - c2.r) * (d - c1.r + c2.r) * (d + c1.r + c2.r));
        return a1 + a2 - a3; 
    }
}

std::tuple<int, int, int> calculateMetrics(const std::vector<std::vector<int>>& groundTruth, const std::vector<std::vector<int>>& detections, double iouThreshold) {
    int tp = 0, fp = 0, fn = groundTruth.size();

    for (const auto& detection : detections) {
        bool matched = false;
        for (const auto& truth : groundTruth) {
            double intersection = circleIntersectionArea({double(detection[0]), double(detection[1]), double(detection[2])}, {double(truth[0]), double(truth[1]), double(truth[2])});
            double union_area = M_PI * (detection[2] * detection[2] + truth[2] * truth[2]) - intersection;
            double iou = intersection / union_area;
            if (iou >= iouThreshold) {
                tp++;
                fn--;
                matched = true;
                break;
            }
        }
        if (!matched) {
            fp++;
        }
    }

    return std::make_tuple(tp, fp, fn);
}

void detectCircles(const Mat& image) {
    Mat no_noise, binary;

    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(image, no_noise, cv::Size(ksize, ksize), 0);
    imwrite("../prj.lab/lab04/output/bluredImage.png", no_noise);


    adaptiveThreshold(no_noise, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 33, 0);
    imwrite("../prj.lab/lab04/output/binImage.png", binary);

    Mat labels, stats, centroids;
    int numLabels = connectedComponentsWithStats(binary, labels, stats, centroids);

    Mat result;
    Mat detections(numCircles * 50, numCircles * 50, CV_8UC1, Scalar(0));

    cvtColor(image, result, COLOR_GRAY2BGR);

    for (int i = 1; i < numLabels; ++i) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < 50) continue;

        int x = stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH) / 2;
        int y = stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT) / 2;
        int radius = (stats.at<int>(i, CC_STAT_WIDTH) + stats.at<int>(i, CC_STAT_HEIGHT)) / 4;
        x_y_r_detection.push_back({x, y, radius});
        circle(detections, Point(x, y), radius, 255, 2);
        circle(result, Point(x, y), radius, Scalar(0, 0, 255), 2);
    }
    imwrite("../prj.lab/lab04/output/stats_detections.png", detections);
    imwrite("../prj.lab/lab04/output/detections.png", result);

    FileStorage fs("../prj.lab/lab04/output/ground_detections.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    fs << "data" << "{";
    fs << "objects" << "[";

    for (auto elem : x_y_r_detection) {
        double iou = 0.0;
        for (auto truth_elem : x_y_r) {
            int dx = elem[0] - truth_elem[0];
            int dy = elem[1] - truth_elem[1];
            int distance_squared = dx * dx + dy * dy;
            if (distance_squared < (elem[2] + truth_elem[2]) * (elem[2] + truth_elem[2])) {
                double intersection = circleIntersectionArea({double(elem[0]), double(elem[1]), double(elem[2])}, {double(truth_elem[0]), double(truth_elem[1]), double(truth_elem[2])});
                double union_area = M_PI * (elem[2] * elem[2] + truth_elem[2] * truth_elem[2]) - intersection;
                iou = intersection / union_area;
            }
        }
        fs << "{";
        fs << "p" << "[";
        fs << elem[0] << elem[1] << elem[2];
        fs << "]";
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << round(iou*100)/100;
        std::string formatted_number = ss.str();
        fs << "iou" << formatted_number;
        fs << "}";
    }

    fs << "]";
    fs << "}";
    fs.release();
}

int main() {
    Mat image = generateImage();
    imwrite("generated_image.png", image);
    generateImageClear();
    detectCircles(image);

    std::tuple<int, int, int> ans = calculateMetrics(x_y_r, x_y_r_detection, 0.4);
    double acc = (double)get<0>(ans) / (double)(get<0>(ans) + get<1>(ans) + get<2>(ans));
    cout << "True Positives: " << get<0>(ans) << "\n";
    cout << "False Positives: " << get<1>(ans) << "\n";
    cout << "False Negatives: " << get<2>(ans) << "\n";
    cout << "tp/sum: " << acc << "\n";

    return 0;
}
