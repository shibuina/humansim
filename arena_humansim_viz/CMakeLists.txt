cmake_minimum_required(VERSION 3.8)
project(arena_humansim_viz)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(arena_humansim_msgs REQUIRED)

set(dependencies
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
  arena_humansim_msgs
)

add_library(${PROJECT_NAME} STATIC src/build_markers.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(${PROJECT_NAME} visualization_msgs geometry_msgs std_msgs arena_humansim_msgs)

add_executable(${PROJECT_NAME}_node src/viz_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})
ament_target_dependencies(${PROJECT_NAME}_node ${dependencies})

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_build_markers test/test_build_markers.cpp)
  target_link_libraries(test_build_markers ${PROJECT_NAME})
endif()

ament_export_dependencies(${dependencies})
ament_package()
