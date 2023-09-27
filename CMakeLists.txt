cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(odometry_4dradar)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# SET(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    geometry_msgs
    nav_msgs
    sensor_msgs
    roscpp
    rospy
    rosbag
    std_msgs
    image_transport
    tf
    radar_ego_velocity_estimator
    fast_gicp
    ld_msgs
)

find_package(PCL 1.8 REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(Ceres REQUIRED)
FIND_PACKAGE(GTSAM REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(
    include
    third_party/ikd-Tree
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
)

catkin_package(
    CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
    DEPENDS Boost PCL
    INCLUDE_DIRS include
)



list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

add_executable(icp4radar src/iterative_closest_point.cpp third_party/ikd-Tree/ikd_Tree.cpp)
add_executable(radar_odometry src/radar_odometry.cpp third_party/ikd-Tree/ikd_Tree.cpp include/scancontext/Scancontext.cpp)

target_link_libraries(icp4radar ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(radar_odometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CERES_LIBRARIES} -lgtsam)
