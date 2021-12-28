sudo apt-get install -y patchelf libosmesa6-dev

mkdir -p $HOME/.mujoco &&
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz &&
  tar -xf mujoco.tar.gz -C $HOME/.mujoco &&
  rm mujoco.tar.gz

echo "export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}" >>$HOME/.bashrc
pip install "mujoco-py<2.2,>=2.1"
python3 -c "import mujoco_py"
