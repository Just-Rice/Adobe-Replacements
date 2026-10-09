local development setup
=======================

Ubuntu 22.04
------------


### Install Python

```bash
sudo apt install python3.11-venv
```

### Install Docker

```bash
sudo apt install docker.io

# Docker requires user group permissions
# This may require a logout to work
sudo groupadd docker
sudo usermod -aG docker ${USER}
newgrp docker

# Test if Docker works
docker run hello-world
```

MacOS
-----

(TODO)

Windows
-------

(Not supported.)


