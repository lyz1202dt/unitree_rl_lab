ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base

FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG UNITREE_RL_LAB_PATH_ARG
ENV UNITREE_RL_LAB_PATH=${UNITREE_RL_LAB_PATH_ARG}

USER root

COPY . ${UNITREE_RL_LAB_PATH}

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install \
    prettytable==3.3.0 hidapi==0.14.0.post2 tensordict==0.8.3 argcomplete

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install \
    --no-build-isolation --no-deps -e ${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab

WORKDIR ${UNITREE_RL_LAB_PATH}
