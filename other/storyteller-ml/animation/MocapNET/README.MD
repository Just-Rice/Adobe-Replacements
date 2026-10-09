# MocapNET V4 Documentation

Welcome to the MocapNET V4 documentation! 🎉 Immerse yourself in the world of advanced motion capture technology with ease.

🔗 **Find the Original Documentation** at [README_ORIGINAL.md](./README_ORIGINAL.md).

This version from the `tag mnet4` branch is sourced directly from the [official MocapNET repository](https://github.com/FORTH-ModelBasedTracker/MocapNET/tree/mnet4).

## 🚀 Getting Started: Developer Installation via docker (For Server Deployment Usage)

To install MocapNET V4 on a Linux system:

1. **Clone the MocapNET Repository**:
   ```sh
   git clone https://github.com/storytold/storyteller-ml/master/animation/MocapNET.git
   ```

2. **Deployment**:
```sh
cd MocapNET/docker
bash build_and_deploy.sh
```

This script will build and run a Docker image, setting up the necessary environment for MocapNET V4.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

### 🚀 Getting Started: Main Installation Process (Run locally)

For both Linux and Windows (via WSL), follow these steps:

1. **Install Dependencies**:
   ```sh
   sudo apt-get install git build-essential cmake libopencv-dev libjpeg-dev libpng-dev libglew-dev libpthread-stubs0-dev
   ```

2. **Clone the MocapNET Repository**:
   ```sh
   git clone https://github.com/storytold/storyteller-ml/master/animation/MocapNET.git
   ```

3. **Navigate to MocapNET Directory**:
   ```sh
   cd MocapNET
   ```

4. **Run the Initialization Script**:
   ```sh
   ./initialize.sh
   ```

This will set up MocapNET V4 on your system, including all necessary dependencies.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Additional Steps for Windows Users

For Windows users, using the Windows Subsystem for Linux (WSL) is recommended:

1. **Enable WSL on Windows**:
   - Open PowerShell as Administrator and execute:
     ```powershell
     dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
     ```
   - Restart your computer when prompted.

2. **Install a Linux Distribution from the Microsoft Store**:
   - Choose and install your preferred Linux distribution (e.g., Ubuntu, Debian, Fedora).

3. **Set Up Your Linux Distribution**:
   - Launch the Linux distribution and complete the account setup.

4. **Update and Upgrade Your Linux Distribution**:
   - Run `sudo apt update && sudo apt upgrade` in the Linux terminal.

For a detailed WSL setup guide, view [How to Set Up WSL](https://www.youtube.com/watch?v=1HzYU2_t3yc) on YouTube.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## 🎬 How to Use MocapNET V4

### Preparing for Usage

Before running MocapNET V4, navigate to the appropriate directory:

```sh
cd src/python/mnet4
```

### Quick Start with a Sample Video

Now you're ready to download and use a sample video:

1. **Download a Sample Video**:
   ```sh
   wget http://ammar.gr/mocapnet/shuffle.webm -O shuffle.webm
   ```

2. **Process the Video**:
   - **For General Use**: Execute the following command to run MocapNET on the downloaded video:
     ```sh
     python3 -m mediapipeHolisticWebcamMocapNET --from shuffle.webm --ik 0.001 99 99 --smooth 60 10 --all --save --plot --headless 2
     ```
   - **For Docker Demo**: In the Docker environment, use this simplified command. Note that `--plot` and `--save` arguments are removed as they are optional debugging arguments (scroll down to a list of available cli args):
     ```sh
     python3 -m mediapipeHolisticWebcamMocapNET --from shuffle.webm --ik 0.001 99 99 --smooth 60 10 --all --headless 2
     ```

### Convert Output BVH

Before running the conversion command, make sure you're in the root directory of MocapNET. You can do this by executing:

```sh
cd ../../..
```

Now, proceed with the BVH conversion:

```sh
./GroundTruthDumper --from dependencies/RGBDAcquisition/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Motions/DAZFriendlyCGSPEED_ZXYAndHandsAxisBigHands.bvh --merge src/python/mnet4/out.bvh dependencies/RGBDAcquisition/opengl_acquisition_shared_library/opengl_depth_and_color_renderer/Motions//mergeDazFriendlyAndAddHead.profile --setPositionRotation 0 0 0 0 0 0 --bvh remade.bvh
```
______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Notes:

- The output bvh file will be stored inside this directory ``MocapNET/src/python/mnet4`` with the filename ``out.bvh``

- The converted bvh file will be stored in the root directory ``MocapNET`` with the filename ``remade.bvh``

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

## 🌟 BVH File Examples

MocapNET provides different types of BVH (BioVision Hierarchy) files to cater to various needs. Below are two specific examples that demonstrate the range of outputs possible with MocapNET.

### Full-Body Estimation Example

The `shuffle_with_whole_body.bvh` file includes a full-body motion capture estimation, encompassing all major body parts, including the face, hands, toes etc. This file is ideal for comprehensive motion analysis.

- **Access the Full-Body BVH File**: [Shuffle with Whole Body BVH](https://github.com/storytold/storyteller-ml/blob/master/animation/MocapNET/exampleOutputs/shuffle_with_whole_body.bvh)

### Minimalistic Version (Currently Buggy)

The `shuffle_without_face_toes.bvh` file represents a more minimalistic version of motion capture data, where face and toes bones are omitted. Note that this version is currently buggy and may have limitations in its use.

- **Access the Minimalistic BVH File**: [Shuffle without Face and Toes BVH](https://github.com/storytold/storyteller-ml/blob/master/animation/MocapNET/exampleOutputs/shuffle_without_face_toes.bvh)

### Using the BVH Files

These example files are beneficial for:

- **Animation and Simulation**: Load these BVH files into animation software to visualize and further process the motion data for example in Blender.
- **Comparison and Learning**: Use these files to compare the differences in data output and structure between full-body and minimalistic motion capture.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

### Command-Line Options Explained

- `--headless`: Skip the live video display. Ideal for headless servers or GUI-less environments.
- `--live`: Launches a live demo. Functionalities may vary.
- `--mt`: Enable multi-threading for better performance on multi-core CPUs.
- `--calib [filename]`: Use a specific calibration file for camera adjustments.
- `--frameskip [number]`: Skip a set number of frames to reduce load or speed up processing.
- `--nnsubsample [number]`: Run the neural network at set intervals (e.g., every 2nd frame).
- `--ik [float int int]`: Set parameters for inverse kinematics optimization.
- `--smooth [float float]`: Apply smoothing to motion capture data.
- `--noik`: Turn off inverse kinematics processing.
- `--aspectCorrection [float]`: Adjust video or image aspect ratio.
- `--noise [float]`: Add noise to input for testing under varied conditions.
- `--size [int int]`: Define video/image dimensions.
- `--scale [float]`: Scale the output motion data.
- `--plot`: Plot BVH channels for skeletal animation visualization.
- `--all`: Activate all processing features (body, eyes, mouth, hands).
- `--nobody`: Disable body joint estimation.
- `--face`: Enable face joint estimation.
- `--eyes / --reye`: Estimate right eye joints.
- `--mouth`: Enable mouth joint estimation.
- `--hands`: Estimate hand joints.
- `--save`: Save outputs (images, videos, data, logs).
- `--engine [engine_name]`: Choose a specific neural network backend.
- `--from [filepath]`: Specify input source (video file or live feed).
- `--profile`: Activate performance profiling.

## THIS README IS A W.I.P 
