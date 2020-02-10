#!/bin/bash
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
CLEAR="\033[0m"

CPU_ARCH=$(uname -m)
OS_VERSION=$(uname -v)

function info {
  echo -e "${GREEN}${1}${CLEAR}"
}

function warn {
  echo -e "${YELLOW}${1}${CLEAR}"
}

function error {
  echo -e "${RED}${1}${CLEAR}"
}

if [[ -f /etc/mendel_version ]]; then
  warn "Looks like you're using a Coral Dev Board. You should instead use Debian packages to manage Edge TPU software."
  exit 0
fi

if [[ "${CPU_ARCH}" == "x86_64" ]] && [[ "${OS_VERSION}" == *"Debian"* || "${OS_VERSION}" == *"Ubuntu"* ]]; then
  info "Recognized as Linux on x86_64."
  HOST_GNU_TYPE=x86_64-linux-gnu
elif [[ "${CPU_ARCH}" == "armv7l" ]]; then
  MODEL=$(cat /proc/device-tree/model)
  if [[ "${MODEL}" == "Raspberry Pi 3 Model B Rev"* ]]; then
    info "Recognized as Raspberry Pi 3 B."
    HOST_GNU_TYPE=arm-linux-gnueabihf
  elif [[ "${MODEL}" == "Raspberry Pi 3 Model B Plus Rev"* ]]; then
    info "Recognized as Raspberry Pi 3 B+."
    HOST_GNU_TYPE=arm-linux-gnueabihf
  fi
elif [[ "${CPU_ARCH}" == "aarch64" ]]; then
  info "Recognized as generic ARM64 platform."
  HOST_GNU_TYPE=aarch64-linux-gnu
fi

if [[ -z "${HOST_GNU_TYPE}" ]]; then
  error "Your platform is not supported. There's nothing to uninstall."
  exit 1
fi

# Device rule file.
UDEV_RULE_PATH="/etc/udev/rules.d/99-edgetpu-accelerator.rules"
if [[ -f "${UDEV_RULE_PATH}" ]]; then
  info "Unnstalling device rule file [${UDEV_RULE_PATH}]..."
  sudo rm -f "${UDEV_RULE_PATH}"
  sudo udevadm control --reload-rules && udevadm trigger
  info "Done."
fi

# Runtime library.
LIBEDGETPU_DST="/usr/lib/${HOST_GNU_TYPE}/libedgetpu.so.1.0"
if [[ -f "${LIBEDGETPU_DST}" ]]; then
  info "Uninstalling Edge TPU runtime library [${LIBEDGETPU_DST}]..."
  sudo rm -f "${LIBEDGETPU_DST}"
  sudo ldconfig
  info "Done."
fi

# Python API.
if sudo python3 -m pip show edgetpu 1>/dev/null; then
  info "Uninstalling Edge TPU Python API..."
  sudo python3 -m pip uninstall -y edgetpu
  info "Done."
fi
