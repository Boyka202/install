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
