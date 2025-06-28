# Copyright (c) 2024，D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.5)
project(hand_lmk_detection)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ai_msgs REQUIRED)
find_package(dnn_node REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(hbm_img_msgs REQUIRED)
add_definitions(-DSHARED_MEM_ENABLED)

include_directories(include
  ${PROJECT_SOURCE_DIR}
)

if(PLATFORM_X3)
  message("build platform X3")
  set(PREFIX_PATH x3)
elseif(PLATFORM_Rdkultra)
  message("build ${PROJECT_NAME} for platform Rdkultra")
  set(PREFIX_PATH rdkultra)
elseif(PLATFORM_X5)
  message("build platform X5")
  set(PREFIX_PATH x5)
elseif(PLATFORM_X86)
  message("build platform X86")
  set(PREFIX_PATH x3)
else()
  message("invalid platform, build platform X3 default")
  set(PREFIX_PATH x3)
endif()

message("CMAKE_SYSROOT is " ${CMAKE_SYSROOT})
include_directories(
  ${CMAKE_SYSROOT}/usr/include/
  ${CMAKE_SYSROOT}/usr/hobot/include/
  ${CMAKE_SYSROOT}/usr/include/opencv/
)

link_directories(
  ${CMAKE_SYSROOT}/usr/lib/hbbpu/
  ${CMAKE_SYSROOT}/usr/hobot/lib/
  ${CMAKE_SYSROOT}/usr/lib/hbmedia/
)

add_executable(${PROJECT_NAME}
  src/main.cpp
  src/hand_lmk_det_node.cpp
  src/hand_lmk_output_parser.cpp
  src/ai_msg_manage.cpp
)

ament_target_dependencies(
  ${PROJECT_NAME}
  rclcpp
  dnn_node
  sensor_msgs
  ai_msgs
  hbm_img_msgs
)

include_directories(include
${PROJECT_SOURCE_DIR}
)

# Install executables
install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  ${PROJECT_SOURCE_DIR}/config/${PREFIX_PATH}/
  DESTINATION lib/${PROJECT_NAME}/config/
)

install(FILES
  ${PROJECT_SOURCE_DIR}/config/960x544.nv12
  DESTINATION lib/${PROJECT_NAME}/config/
)

install(DIRECTORY
${PROJECT_SOURCE_DIR}/launch/
DESTINATION share/${PROJECT_NAME}/launch)

ament_package()