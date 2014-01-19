cmake_minimum_required(VERSION 2.8)
project( opencv-exercises )
find_package( OpenCV REQUIRED )

add_executable( SVMTest SVMTest.cpp )
target_link_libraries( SVMTest ${OpenCV_LIBS} )
