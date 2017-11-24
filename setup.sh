#!/bin/sh

RUN_CMAKE=false
RUN_MAKE=false
RUN_SET_PATHS=false
QUIET=false

usage() {
    echo "USAGE: source setup.sh [options]"
    echo "OPTIONS:"
    echo "\t--help                 -h: Display help"
    echo "\t--set-paths            -p: Setup the enviroment path"
    echo "\t--cmake                -c: Run CMake"
    echo "\t--make                 -m: Run make"
    echo "\t--quiet                -q: Set quiet option"
    echo "\t--all                  -a: Run all the settings"
}

# Parse through the arguments and check if any relavant flag exists
while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        -p | --set-paths)
            RUN_SET_PATHS=true
            ;;
        -c | --cmake)
            RUN_CMAKE=true
            ;;
        -m | --make)
            RUN_MAKE=true
            ;;
        -a | --all)
            RUN_SET_PATHS=true
            RUN_CMAKE=true
            RUN_MAKE=true
            ;;
        -q | --quiet)
            QUIET=true
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            return 1
            ;;
    esac
    shift
done

# If build does not exist create one
mkdir -p build
cd build

if $RUN_SET_PATHS
then
    # Set the enviroment
    echo "Setting up the enviroment paths"
    export PATH=$PATH:/vol/project/2017/362/g1736211/
    source /vol/cuda/8.0.44/setup.sh
    export CUDA_HOME=$CUDA_PATH
    echo "CUDA_HOME set to $CUDA_HOME"
fi

if $RUN_CMAKE
then
    echo "Running CMake ..."
    if ! $QUIET
    then
        cmake -DCMAKE_PREFIX_PATH=/vol/project/2017/362/g1736211/share/OpenCV .. || (cd ../ && return 1)
    else
        cmake -DCMAKE_PREFIX_PATH=/vol/project/2017/362/g1736211/share/OpenCV .. >/dev/null 2>&1 || (cd ../ && return 1)
    fi
fi

if $RUN_MAKE
then
    echo "Running make ..."
    if ! $QUIET
    then
        make -j4 || (cd ../ && return 1)
    else
        make -j4 >/dev/null 2>&1 || (cd ../ && return 1)
    fi
    echo "Make complete!"
fi
cd ..
