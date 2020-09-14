include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT "/usr/local/cuda/include" CACHE PATH "cuDNN root folder")

find_path(CUDNN_INCLUDE cudnn.h
  PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT}
  DOC "Path to cuDNN include directory." )

find_library(CUDNN_LIBRARY NAMES libcudnn.so cudnn.lib # libcudnn_static.a
  PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE}
  PATH_SUFFIXES lib lib/x64  cuda/lib cuda/lib64 lib/x64
  DOC "Path to cuDNN library.")

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARY CUDNN_INCLUDE)

mark_as_advanced(CUDNN_ROOT CUDNN_INCLUDE CUDNN_LIBRARY)