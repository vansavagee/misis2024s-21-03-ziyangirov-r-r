#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv; 
using namespace std; 
int gammaCorrection(int color, double gamma) {
    return (int)(pow((double)color / 255, gamma) * 255);
}
int main(int argc, char** argv) {

    CommandLineParser parser(argc, argv,
        "{@imageName    |      | output image name}"
        "{s             | 3    | width of one peace}"
        "{h             | 30   | height of one peace}"
        "{gamma         | 2.4  | coef for gamma correction}"
    );
    
    String name = parser.get<String>("@imageName");
    int s = parser.get<int>("s");
    int h = parser.get<int>("h");
    double gamma = parser.get<double>("gamma");

    Mat1b m(2 * h, s * 256, 1);
    for (int step = 0; step < 256; step++) 
        for (int c = s * step; c < s * (step + 1); c++) 
            for (int r = 0; r < 2 * h; r++) 
                r <= h ? m[r][c] = step :  m[r][c] = gammaCorrection(step, gamma);  
    
    imshow(name, m);
    waitKey(0);
    return 0;
}