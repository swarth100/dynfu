file(GLOB SRCS *.cpp)

ADD_EXECUTABLE(dualQuaternionTest ${SRCS})

TARGET_LINK_LIBRARIES(dualQuaternionTest
  libgtest
  libgmock
)

install(TARGETS dualQuaternionTest DESTINATION test)
