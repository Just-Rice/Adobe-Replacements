#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

ORIG_DIR=`pwd`
  
ASK_QUESTIONS=1
if [[ $* == *--collab* ]]
then 
 echo "Using non-blocking collab mode"
 ASK_QUESTIONS=0
fi

if [ "$ASK_QUESTIONS" -eq "0" ]; then
   echo "We assume collab machines have everything we need.."
else
# Simple dependency checker that will apt-get stuff if something is missing
# sudo apt-get install build-essential cmake libopencv-dev libjpeg-dev libpng-dev libglew-dev libpthread-stubs0-dev
SYSTEM_DEPENDENCIES="wget git build-essential cmake libopencv-dev libjpeg-dev libpng-dev libglew-dev libpthread-stubs0-dev"
#------------------------------------------------------------------------------
for REQUIRED_PKG in $SYSTEM_DEPENDENCIES
do
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo "Checking for $REQUIRED_PKG: $PKG_OK"
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get install $SYSTEM_DEPENDENCIES  
  break
fi
done
#------------------------------------------------------------------------------
fi

# Downloading Tensorflow 2.3.1 with GPU support
echo "Selected Tensorflow version gpu/2.3.1"
cd "$DIR"
if [ -f /usr/local/lib/libtensorflow.so ]; then
 echo "Found a system wide tensorflow installation, not altering anything"
elif [ -f dependencies/libtensorflow/lib/libtensorflow.so ]; then
 echo "Found a local tensorflow installation, not altering anything"
else 
 echo "Tensorflow not found, downloading Tensorflow 2.3.1 with GPU support.."
 if [ ! -f dependencies/libtensorflow-gpu-linux-x86_64-2.3.1.tar.gz ]; then
   cd "$DIR/dependencies"
   wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.3.1.tar.gz
 else
   echo "Tensorflow 2.3.1 GPU tarball already downloaded."
 fi
fi

cd "$DIR"
if [ -f dependencies/libtensorflow-gpu-linux-x86_64-2.3.1.tar.gz ]; then
 #Doing a local installation that requires no SUDO 
 cd "$DIR/dependencies"
 mkdir libtensorflow
 tar -C libtensorflow -xzf libtensorflow-gpu-linux-x86_64-2.3.1.tar.gz  
 #echo "Please give me sudo permissions to install Tensorflow $TENSORFLOW_VERSION C Bindings.."
 #sudo tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-$TENSORFLOW_VERSION.tar.gz
 else
   echo "Failed to download/extract tensorflow.."
fi
#---------------------------------------------------------------------------------------------------------------------------

cd "$DIR"
if [ -f dependencies/RGBDAcquisition/README.md ]; then
 echo "RGBDAcquisition appears to already exist .."
else
 cd "$DIR/dependencies"
 git clone https://github.com/AmmarkoV/RGBDAcquisition
 cd RGBDAcquisition
 mkdir build
 cd build
 cmake ..
 cd ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer
 mkdir build
 cd build
 cmake ..
 cd "$DIR"
fi

cd "$DIR"
cd src/python/mnet4
rm -rf BVH/
ln -s ../../../dependencies/RGBDAcquisition/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Applications/BVHTester/ BVH
cd BVH
./makeLibrary.sh

cd ..
# Install rest of python stuff
./setup.sh

echo "Now to try and build MocapNET.."
cd "$DIR"

if [ -d "build" ]; then
  echo "Build directory already exists, but since the initialize.sh script was called "
  echo "this means the user wants to also initialize the build directory, so doing this"
  rm -rf build/
fi

mkdir build
cd build
cmake ..
make 
cd "$DIR"

exit 0
