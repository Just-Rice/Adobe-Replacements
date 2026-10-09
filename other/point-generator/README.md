point-generator
===============
Generate points to send over ZeroMQ to the Unreal Engine point cloud demo.

Windows installation
--------------------
The best solution is to depend on the provided vendored libs:

```toml
zmq = { version = "0.9", features = ['vendored'] }
```

Trying to build our own zeromq is COMPLICATED. I don't understand the Windows ecosystem at all, 
and I spent hours trying to get things working:

* Install [Microsoft's vcpkg](https://github.com/Microsoft/vcpkg/) package manager.
* `PS C:\dev\vcpkg> .\vcpkg.exe install czmq`
* [Install CMake](https://cmake.org/download/) and add the bin directory (`C:\Program Files\CMake\bin`)
  to the path (System properties > Advanced > Environment variables).
  
* Build:

```
mkdir build
cd build
cmake ..
```

* Then open the `.sln` file in Visual Studio and build... 
* which gives you `dll` and `lib` files...
* Add those to the windows Path...
* (Not working)

