# install
sudo apt update
sudo apt install git cmake build-essential libfftw3-dev libhdf5-dev python3 python3-pip
git clone https://github.com/DAMASK-Project/damask.git
cd damask
mkdir build
cd build
cmake ..
make -j$(nproc)
export PATH=$PATH:/path/to/damask/build/bin
pip3 install -r requirements.txt
pip3 install .
damask --help


sudo apt install git cmake build-essential libfftw3-dev libhdf5-dev python3-pip
git clone https://github.com/DAMASK-Project/damask.git
cd damask
mkdir build && cd build
cmake ..
make -j$(nproc)


sudo nano /etc/apt/sources.list
http://archive.ubuntu.com/ubuntu
http://security.ubuntu.com/ubuntu
http://old-releases.ubuntu.com/ubuntu
deb http://old-releases.ubuntu.com/ubuntu lunar main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu lunar-updates main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu lunar-backports main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu lunar-security main restricted universe multiverse
sudo apt update
sudo apt upgrade
sudo do-release-upgrade -d


sudo add-apt-repository ppa:damask/ppa
sudo apt update
sudo apt install damask

ERROR: ppa 'damask/ppa' not found (use --login if private)

ddddddd
sudo bash -c 'cat > /etc/apt/sources.list <<EOF
deb http://old-releases.ubuntu.com/ubuntu lunar main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu lunar-updates main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu lunar-security main restricted universe multiverse
deb http://old-releases.ubuntu.com/ubuntu lunar-backports main restricted universe multiverse
EOF'

sudo apt update
sudo apt upgrade

sudo do-release-upgrade -d

git clone https://github.com/damask-msc/damask.git
cd damask

sudo apt install -y cmake gfortran build-essential libfftw3-dev libhdf5-dev liblapack-dev libblas-dev python3 python3-pip

mkdir build
cd build
cmake ..
make -j$(nproc)


damask --version


