#!/bin/bash

set -ex
DIR=`pwd`
cd "$DIR"

ORIG_DIR=`pwd`

#Simple dependency checker that will apt-get stuff if something is missing
# sudo apt-get install build-essential cmake libopencv-dev libjpeg-dev libpng-dev libglew-dev libpthread-stubs0-dev
SYSTEM_DEPENDENCIES="wget git build-essential cmake libopencv-dev libjpeg-dev libpng-dev libglew-dev libpthread-stubs0-dev"
#------------------------------------------------------------------------------
for REQUIRED_PKG in $SYSTEM_DEPENDENCIES
do
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo "Checking for $REQUIRED_PKG: $PKG_OK"
if [ "" = "$PKG_OK" ]; then

  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."

  #If this is uncommented then only packages that are missing will get prompted..
  #sudo apt-get --yes install $REQUIRED_PKG

  #if this is uncommented then if one package is missing then all missing packages are immediately installed..
  sudo apt-get install $SYSTEM_DEPENDENCIES  
  break
fi
done
#------------------------------------------------------------------------------





cd "$DIR"
if [ -f dependencies/RGBDAcquisition/README.md ]; then
 echo "RGBDAcquisition appears to already exist .."
 #cd "$DIR/dependencies/RGBAcquisition"
 #echo "We can make sure that it is up to date to avoid issues like https://github.com/FORTH-ModelBasedTracker/MocapNET/issues/48.."
 #echo "However someone might not want to update the code.. what to do.."
 #git pull
 #cd "$DIR"
else
 cd "$DIR/dependencies"
 git clone https://github.com/AmmarkoV/RGBDAcquisition
 cd RGBDAcquisition
 # This package has no releases so we just picked a "good" checkout
 git checkout b752895
 mkdir build
 cd build
 cmake ..
 #We dont need to make it 
 #make 
 cd ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer
 mkdir build
 cd build
 cmake ..
 #We dont need to make it 
 #make 
 cd "$DIR"
 #Also retrieve Renderer 
 #ln -s dependencies/RGBDAcquisition/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Renderer 
fi

if [ -f /usr/local/lib/libtensorflow.so ]; then
 echo "Found a system wide tensorflow installation, not altering anything"
elif [ -f dependencies/libtensorflow/lib/libtensorflow.so ]; then
 echo "Found a local tensorflow installation, not altering anything"
else 
 echo "Tensorflow not found, downloading Tensorflow 2.3.1 with GPU support.."
 if [ ! -f dependencies/libtensorflow-gpu-linux-x86_64-2.3.1.tar.gz ]; then
   cd "$DIR/dependencies"
   wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.3.1.tar.gz
    # mkdir libtensorflow
#  tar -C libtensorflow -xzf   
 #echo "Please give me sudo permissions to install Tensorflow $TENSORFLOW_VERSION C Bindings.."
 tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.3.1.tar.gz
 else
   echo "Tensorflow 2.3.1 GPU tarball already downloaded."
 fi
fi


cd "$DIR"
cd src/python/mnet4
rm -rf BVH/
ln -s ../../../dependencies/RGBDAcquisition/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Applications/BVHTester/ BVH
cd BVH
./makeLibrary.sh

cd ..
#Install rest of python stuff..
./setup.sh


 
 
#Now that we have everything lets build..
echo "Now to try and build MocapNET.."
cd "$DIR"

#if there is an already existing build directory and you called initialize.sh
#it will get initialized as well.. (https://github.com/FORTH-ModelBasedTracker/MocapNET/issues/19)
if [ -d "build" ]; then
  echo "Build directory already exists, but since the initialize.sh script was called "    
  echo "this means the user wants to also initialize the build directory, so doing this"    
  echo "to prevent problems like https://github.com/FORTH-ModelBasedTracker/MocapNET/issues/19"   
  rm -rf build/
fi

mkdir build
cd build
cmake ..
make 
cd "$DIR"
 


exit 0
