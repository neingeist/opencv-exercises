cmake_minimum_required(VERSION 2.8)
project( opencv-exercises )
find_package( OpenCV REQUIRED )

add_executable( SVMTest SVMTest.cpp )
target_link_libraries( SVMTest ${OpenCV_LIBS} )

add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

add_executable( SharpenImage SharpenImage.cpp )
target_link_libraries( SharpenImage ${OpenCV_LIBS} )
