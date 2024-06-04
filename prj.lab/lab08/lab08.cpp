#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

int main(int argc, char *argv[]) {
    cv::Mat img = cv::imread("../prj.lab/lab08/input/italy.jpg", cv::IMREAD_COLOR);

    cv::Mat bgr_channels[3];
    cv::split(img, bgr_channels);
    cv::Mat blue_channel = bgr_channels[0];
    cv::Mat green_channel = bgr_channels[1];
    cv::Mat red_channel = bgr_channels[2];

    cv::Mat projection = (cv::Mat_<double>(2, 3) << 
        0.5 + 0.5 / sqrt(3), -0.5 + 0.5 / sqrt(3), -1 / sqrt(3),
        -0.5 + 0.5 / sqrt(3), 0.5 + 0.5 / sqrt(3), -1 / sqrt(3));

    cv::Mat rotation = (cv::Mat_<double>(2, 2) <<
        (sqrt(6) + sqrt(2)) / 4, -(sqrt(6) - sqrt(2)) / 4,
        (sqrt(6) - sqrt(2)) / 4, (sqrt(6) + sqrt(2)) / 4);

    std::vector<cv::Vec2f> projected_points;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            cv::Mat point_mat = (cv::Mat_<double>(3, 1) << (double)pixel[2] / 255.0, (double)pixel[1] / 255.0, (double)pixel[0] / 255.0);
            cv::Mat result;
            cv::gemm(projection, point_mat, 1.0, cv::Mat(), 0.0, result);
            cv::gemm(rotation, result, 1.0, cv::Mat(), 0.0, result);

            projected_points.push_back(cv::Vec2f(result.at<double>(0, 0), result.at<double>(1, 0)));
        }
    }

    int image_size = 512;
    cv::Mat plot_image(image_size, image_size, CV_8UC3, cv::Scalar(255, 255, 255));

    std::vector<cv::Point> hexagon_points;
    std::vector<cv::Scalar> hexagon_colors;
    int hex_size = image_size / 3;  
    int center_x = image_size / 2;
    int center_y = image_size / 2;

    cv::Scalar colors[6] = {
        cv::Scalar(0, 0, 255),  
        cv::Scalar(0, 127, 127), 
        cv::Scalar(0, 255, 0),   
        cv::Scalar(127, 127, 0), 
        cv::Scalar(255, 0, 0),  
        cv::Scalar(127, 0, 127) 
    };

    for (int i = 0; i < 6; i++) {
        double angle = CV_PI / 3.0 * i; 
        int x = static_cast<int>(center_x + hex_size * cos(angle));
        int y = static_cast<int>(center_y + hex_size * sin(angle));
        hexagon_points.push_back(cv::Point(x, y));
        hexagon_colors.push_back(colors[i]);
    }

    for (size_t i = 0; i < hexagon_points.size(); i++) {
        cv::Point p1 = hexagon_points[i];
        cv::Point p2 = hexagon_points[(i + 1) % hexagon_points.size()];
        cv::Scalar color1 = hexagon_colors[i];
        cv::Scalar color2 = hexagon_colors[(i + 1) % hexagon_colors.size()];

        int num_steps = 100; 
        for (int step = 0; step <= num_steps; step++) {
            double alpha = (double)step / num_steps;
            cv::Point interp_point = p1 + alpha * (p2 - p1);
            cv::Scalar interp_color = color1 * (1 - alpha) + color2 * alpha;
            cv::circle(plot_image, interp_point, 1, interp_color, -1);
        }
    }

    cv::Mat density_map = cv::Mat::zeros(image_size, image_size, CV_32F);

    double scale_factor = hex_size;
    for (const auto& point : projected_points) {
        int x = static_cast<int>(point[0] * scale_factor + center_x);
        int y = static_cast<int>(point[1] * scale_factor + center_y);
        if (x >= 0 && x < image_size && y >= 0 && y < image_size) {
            density_map.at<float>(y, x) += 1.0f;
        }
    }

    cv::normalize(density_map, density_map, 0, 255, cv::NORM_MINMAX);
    density_map.convertTo(density_map, CV_8UC1);

    cv::Mat density_colormap;
    cv::applyColorMap(density_map, density_colormap, cv::COLORMAP_JET);

    for (int y = 0; y < image_size; ++y) {
        for (int x = 0; x < image_size; ++x) {
            if (density_map.at<uchar>(y, x) > 0) {
                plot_image.at<cv::Vec3b>(y, x) = density_colormap.at<cv::Vec3b>(y, x);
            }
        }
    }

    cv::imwrite("../prj.lab/lab08/output/italy.png", plot_image);

    return 0;
}
