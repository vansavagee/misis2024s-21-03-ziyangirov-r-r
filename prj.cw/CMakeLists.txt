add_executable(coursework main.cpp)

target_link_libraries(coursework PRIVATE opencv_core opencv_imgproc opencv_highgui nlohmann_json::nlohmann_json)