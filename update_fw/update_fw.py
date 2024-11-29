#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from radardemoinfra.eth_devices import EthDevices
from radardemoinfra.fpga_serial import FpgaSerial

SSH_PUB_KEY = Path(os.environ["HOME"]) / ".ssh" / "id_ed25519.pub"
SSH_PRV_KEY = Path(os.environ["HOME"]) / ".ssh" / "id_ed25519"


def create_ssh_key():
    print("Create new key")
    _ = os.system(f"ssh-keygen -f {SSH_PRV_KEY} -t ed25519 -N ''")


def read_public_key() -> str:
    print("Reading public key")
    with open(SSH_PUB_KEY, "r") as pk:
        return pk.read()


def get_public_key() -> str:
    if not SSH_PUB_KEY.exists():
        create_ssh_key()
    return read_public_key()


def target() -> str:
    fpga = FpgaSerial()
    fpga.append_authorized_key(get_public_key())
    eth_devices = EthDevices()
    target = f"root@'[{fpga.get_ipv6()}%{eth_devices.usb}]':/boot/."
    print(f"Target = {target}")
    return target


def copy_files(dir: Path, target: str):
    file_list = [p for p in Path(dir).iterdir() if p.is_file()]
    for file in file_list:
        print(
            f"scp -6 -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null {file} {target}"
        )
        _ = os.system(
            f"scp -6 -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null {file} {target}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the fpga firmware")
    _ = parser.add_argument(
        "-s",
        "--source",
        help="The source directory of the files to be copied to the card",
        default="hw",
    )
    args = parser.parse_args()
    if not Path(args.source).exists():
        print(f"ERROR: Path {args.source} does not exist")
        exit(1)
    copy_files(Path(args.source), target())
