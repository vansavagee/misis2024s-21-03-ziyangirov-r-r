#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat1f genTestImage() {
    int squareSide = 99, radius = 25, black = 0, gray = 127, white = 255;
    Mat1f result(99 * 2, 99 * 3);
    vector<Vec2b> colors { Vec2b{0, 127}, Vec2b{127, 0}, Vec2b{255, 0}, Vec2b{255, 127}, Vec2b{0, 255}, Vec2b{127, 255}  };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            Vec2b color = colors[j * 3 + i];
            Point top_left = Point(i * squareSide, j * squareSide);
            Point bottom_right = Point((i + 1) * squareSide, (j + 1) * squareSide);
            rectangle(result, top_left, bottom_right, color[0], FILLED);
            Point center = Point(i * squareSide + squareSide / 2., j * squareSide + squareSide / 2.);
            ellipse(result, center, Size(radius, radius), 0, 0, 360, color[1], FILLED);
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    Mat1f testImg = genTestImage();

    Mat core1 = (Mat1f(2, 2) << 1.0, 0.0, 0.0, -1.0);
    Mat core2 = (Mat1f(2, 2) << 0.0, 1.0, -1.0, 0.0);
    Mat1f I1, I2;
    filter2D(testImg, I1, -1, core1);
    filter2D(testImg, I2, -1, core2);
    I1 = I1 * 0.5 +  127.5;
    I2 = I2 * 0.5 +  127.5;
    Mat1f I3 = testImg.clone(); 

    for (int r = 0; r < testImg.rows; r++) {
        for (int c = 0; c < testImg.cols; c++) 
            I3[r][c] = sqrt(I1[r][c] * I1[r][c] + I2[r][c] * I2[r][c]);
    }

    Mat1b I1_b = I1.clone(), I2_b = I2.clone(), I3_b = I3.clone();
    vector<Mat1b> threeParts{I1_b, I2_b, I3_b};
    Mat3b result;
    merge(threeParts, result);

    imwrite("../prj.lab/lab05/output/testImage.jpg", testImg);
    imwrite("../prj.lab/lab05/output/I1.jpg", I1_b);
    imwrite("../prj.lab/lab05/output/I2.jpg", I2_b);
    imwrite("../prj.lab/lab05/output/I3.jpg", I3_b);
    imwrite("../prj.lab/lab05/output/result.jpg", result);
}