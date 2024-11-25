#!/usr/bin/env python3
import re
import subprocess

from radardemoinfra.eth_devices import EthDevices
from radardemoinfra.fpga_serial import FpgaSerial


def get_ipv4_address(device: str) -> str:
    result = subprocess.run(
        ["ip", "addr", "show", "dev", device],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = result.stdout.split("\n")
    pattern = re.compile(
        "\s*inet (?P<ipv4>\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}).*"  # pyright: ignore [reportInvalidStringEscapeSequence]
    )
    for line in lines:
        if m := pattern.match(line):
            return m.group("ipv4")

    # No match found
    return ""


if __name__ == "__main__":
    eth_devices = EthDevices()
    ipv4_address = get_ipv4_address(eth_devices.pcie)
    fpga = FpgaSerial()
    ipv6_address = fpga.get_ipv6()
    result = subprocess.run(
        [
            "socat",
            "-d",
            f"TCP4-LISTEN:2222,bind={ipv4_address},fork",
            f"TCP6:'[{ipv6_address}%{eth_devices.usb}]':22",
        ]
    )
