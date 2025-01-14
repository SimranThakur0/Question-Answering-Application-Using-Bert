echo [$(date)]: "START"
echo [$(date)]: "Creating env with python 3.10 version"
conda create --prefix ./env python=3.10 -y
echo [$(date)]: "activating the env"
source activate ./env
echo [$(date)]: "installing required packages"
pip install -r requirements.txt
echo [$(date)]: "END"