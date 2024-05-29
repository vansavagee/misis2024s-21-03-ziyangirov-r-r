#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <random>
#include <vector>

using namespace cv; 
using namespace std; 

Mat1b drawImage(int black, int gray, int white) {
    Mat1b image(256, 256, black);
    Point center(image.cols / 2, image.rows / 2);
    int squareSide = 209;
    int x = center.x - squareSide / 2;
    int y = center.y - squareSide / 2;

    Rect squareRect(x, y, squareSide, squareSide);
    rectangle(image, squareRect, gray, -1);
    circle(image, center, 83, white, -1);

    return image;
}

Mat1b drawHistogram(Mat1b& image) {
    Mat1b histogram(256, 256, 230);
    int hist[256] = {0};
    float maxVal = 0;

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            int intensity = image(i, j);
            hist[intensity]++;
            if (hist[intensity] > maxVal) {
                maxVal = hist[intensity];
            }
        }
    }

    float scale = static_cast<float>(230) / maxVal;

    for (int i = 0; i < 256; ++i) {
        Point start_point(i, 256);
        Point end_point(i, 256 - hist[i] * scale);
        line(histogram, start_point, end_point, 0, 1);
    }

    return histogram;
}

void addNoise(Mat1b& image, double stdDev) {
    Mat noisyImg = image.clone();
    RNG rng; 
    Mat noise(noisyImg.size(), CV_64FC1);
    rng.fill(noise, RNG::NORMAL, 0, stdDev);
    noisyImg += noise;
    image = noisyImg;
}

int main(int argc, char** argv) {
    Mat1b ans[3];
    vector<int> sigma = {3, 7, 15};
    int colors[4][3] = {{0,127,255}, {20, 127, 235}, {55, 127, 200}, {90, 127, 165}};
    Mat1b image[4];
    Mat1b histogram[4];
    for(int j = 0; j < 3; ++j){
        for(int i = 0; i < 4; ++i){
            image[i] = drawImage(colors[i][0],colors[i][1],colors[i][2]);
            addNoise(image[i], sigma[j]);
            histogram[i] = drawHistogram(image[i]);
            if(i != 0){
                hconcat(image[0], image[i], image[0]);
                hconcat(histogram[0], histogram[i], histogram[0]);
            }
        }
        vconcat(image[0], histogram[0], image[0]);
        ans[j] = image[0];
    }
    
    imshow("Histogram sigma = 3", ans[0]);
    imshow("Histogram sigma = 7", ans[1]);
    imshow("Histogram sigma = 15", ans[2]);
    waitKey(0);
    return 0;
}