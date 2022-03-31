#!/bin/bash

sudo apt-get install -y patchelf libosmesa6-dev

[ ! -d "${HOME}/.mujoco" ] && mkdir -p "${HOME}/.mujoco"

if [ ! -d "${HOME}/.mujoco/mujoco210" ]; then
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz &&
    tar -xf mujoco.tar.gz -C "${HOME}/.mujoco" &&
    rm mujoco.tar.gz
  echo "export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}" >>"${HOME}/.bashrc"
fi
