#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace cv;
using namespace std;
using json = nlohmann::json;

struct Box {
    int x, y, width, height;
};

void calculateMetrics(int TP, int FP, int FN) {
    double precision = TP / double(TP + FP);
    double recall = TP / double(TP + FN);
    double f1 = 2 * (precision * recall) / (precision + recall);
    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "F1-score: " << f1 << endl;
}

float computeIoU(const Rect& box1, const Rect& box2) {
    int x1 = max(box1.x, box2.x);
    int y1 = max(box1.y, box2.y);
    int x2 = min(box1.x + box1.width, box2.x + box2.width);
    int y2 = min(box1.y + box1.height, box2.y + box2.height);

    int intersectionArea = max(0, x2 - x1) * max(0, y2 - y1);
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;

    return static_cast<float>(intersectionArea) / (box1Area + box2Area - intersectionArea);
}

vector<Box> loadGroundTruth(const string& filePath) {
    ifstream inFile(filePath);
    if (!inFile.is_open()) {
        cerr << "Error: Unable to open ground truth file." << endl;
        return {};
    }
    json groundTruthJson;
    inFile >> groundTruthJson;

    vector<Box> groundTruth;
    for (const auto& item : groundTruthJson) {
        groundTruth.push_back({item["x"], item["y"], item["width"], item["height"]});
    }
    return groundTruth;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <image_path> <ground_truth_json>" << endl;
        return -1;
    }

    Mat image = imread(argv[1]);
    if (image.empty()) {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    vector<Box> groundTruth = loadGroundTruth(argv[2]);
    if (groundTruth.empty()) {
        cerr << "Error: No ground truth data loaded." << endl;
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

    Mat result1 = image.clone();
    Mat result = image.clone();
    int TP = 0, FP = 0, FN = 0;

    vector<Rect> detectedBoxes;
    for (const auto& contour : contours) {
        Rect boundingBox = boundingRect(contour);
        float aspectRatio = (float)boundingBox.width / boundingBox.height;
        rectangle(result1, boundingBox, Scalar(0, 255, 0), 2);

        if (aspectRatio > 0.2 && aspectRatio < 5.0 && boundingBox.area() > 100) {
            rectangle(result, boundingBox, Scalar(0, 255, 0), 2);
            detectedBoxes.push_back(boundingBox);
        }
    }

    vector<bool> matched(groundTruth.size(), false);
    for (const auto& detected : detectedBoxes) {
        bool isTruePositive = false;
        for (size_t i = 0; i < groundTruth.size(); ++i) {
            if (!matched[i] && computeIoU(detected, Rect(groundTruth[i].x, groundTruth[i].y, groundTruth[i].width, groundTruth[i].height)) > 0.5) {
                TP++;
                matched[i] = true;
                isTruePositive = true;
                break;
            }
        }
        if (!isTruePositive) {
            FP++;
        }
    }

    FN = count(matched.begin(), matched.end(), false);
    calculateMetrics(TP, FP, FN);

    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Blurred Image", WINDOW_NORMAL);
    namedWindow("Processed Image", WINDOW_NORMAL);
    namedWindow("Detected before filtering Text Areas", WINDOW_NORMAL);
    namedWindow("Detected Text Areas", WINDOW_NORMAL);

    imshow("Original Image", image);
    imshow("Blurred Image", blurred);
    imshow("Processed Image", thresholded);
    imshow("Detected before filtering Text Areas", result1);
    imshow("Detected Text Areas", result);

    waitKey(0);
    return 0;
}
