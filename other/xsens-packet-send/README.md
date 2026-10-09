xsens-packet-send
=================

Since XSENS software is stupidly expensive and restrictive, we've written this piece of software to send 
network packets to Unreal Engine. It cuts out the need for running MVN Animate Pro (which costs thousands 
of dollars a year).

Windows installation
--------------------

1. [Install the Rust programming language compiler](https://www.rust-lang.org/tools/install).
   On Windows, you'll probably want 64-bit for modern machines.

2. The Rust compiler will need Microsoft's C++ compiler and linker.
   [This Microsoft guide](https://docs.microsoft.com/en-us/windows/dev-environment/rust/setup) explains how
   to install support along with Visual Studio.

Running the program on Windows
------------------------------

1. Open `cmd` (command line) or `PowerShell.exe` using the run menu. 

2. Use `cd` ("change directory") to find the location of the program. 
   `dir` can be used to see the current directory contents.
   eg., `cd C:\Path\To\Program`

4. Type `cargo run` to run the program in debug mode (fast compile, slow execution). 
   Ctrl-C will kill the program without needing to close the terminal.

5. Type `cargo build --release` to build a production `.exe` binary file that will run
   six times faster. The exe file will be under the subdirectory `target/release/`.

Program options
---------------

By default, the program plays packets to localhost from "three_dancers_animation.pcap", 
but it can read other files and send to other machines as well.

You can supply arguments to change the recorded packet capture animation:

* Debug: `cargo run -- --file name_of_animation.pcap`
* Release build: `xsens-packet-send.exe --file name_of_animation.pcap`

You can also change the computer you target:

* Debug: `cargo run -- --host 192.168.2.100:8000`
* Release build: `xsens-packet-send.exe --host 192.168.2.100:8000`

You can combine both arguments together:

* Debug: `cargo run -- --file animation.pcap --host 192.168.2.100:8000`
* Release build: `xsens-packet-send.exe --file animation.pcap --host 192.168.2.100:8000`
