TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp

INCLUDEPATH += usr/include\
               usr/include/opencv \
               usr/include/opencv2
LIBS += /usr/lib/x86_64-linux-gnu/libopencv_*.so
