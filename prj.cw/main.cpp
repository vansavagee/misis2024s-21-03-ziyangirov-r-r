#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void calculateMetrics(int TP, int FP) {
    double precision = TP / double(TP + FP);
    cout << "Precision: " << precision << endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    Mat image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    Mat thresholded;
    adaptiveThreshold(blurred, thresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(thresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat result = image.clone();
    int TP = 0, FP = 0; 

    for (const auto& contour : contours) {
        Rect boundingBox = boundingRect(contour);
        float aspectRatio = (float)boundingBox.width / boundingBox.height;

        if (aspectRatio > 0.2 && aspectRatio < 5.0 && boundingBox.area() > 100) {
            rectangle(result, boundingBox, Scalar(0, 255, 0), 2);
            TP++; 
        } else {
            FP++;
        }
    }

    calculateMetrics(TP, FP);

    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Processed Image", WINDOW_NORMAL);
    namedWindow("Detected Text Areas", WINDOW_NORMAL);

    imshow("Original Image", image);
    imshow("Processed Image", thresholded);
    imshow("Detected Text Areas", result);

    waitKey(0);
    return 0;
}
