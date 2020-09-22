set_property(SOURCE ${CMAKE_CURRENT_LIST_DIR}/../../include/mxnet-cpp/op.h PROPERTY GENERATED 1)

add_executable(imagenet_inference "imagenet_inference.cpp")
target_link_libraries(imagenet_inference mxnet_cpp)