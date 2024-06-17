cmake_minimum_required(VERSION 3.10)

# Название проекта
project(ImageRecognition)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Указание пути к заголовочным файлам OpenCV
include_directories(libs/include)

# Указание пути к библиотекам OpenCV
link_directories(libs/lib)

# Добавление исходного файла
add_executable(ImageRecognition WIN32 ImageRecognition.cpp)

# Определение, является ли сборка отладочной или релизной
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OpenCV_LIB opencv_world490d)  # Для отладочной сборки
else()
    set(OpenCV_LIB opencv_world490)   # Для релизной сборки
endif()

# Связывание с библиотекой OpenCV
target_link_libraries(ImageRecognition ${OpenCV_LIB})