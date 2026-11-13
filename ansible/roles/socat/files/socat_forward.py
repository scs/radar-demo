#!/opt/radardemoinfra/venv/bin/python3
import re
import subprocess
import time

from radardemoinfra.eth_devices import EthDevices


def get_ipv4_address(device: str) -> str:
    result = subprocess.run(
        ["ip", "addr", "show", "dev", device],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = result.stdout.split("\n")
    pattern = re.compile(
        r"\s*inet (?P<ipv4>\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}).*"
    )
    for line in lines:
        if m := pattern.match(line):
            return m.group("ipv4")

    # No match found
    return ""


def wait_for_ipv4_address(device: str, attempts: int = 10, delay: float = 1.0) -> str:
    for _ in range(attempts):
        if address := get_ipv4_address(device):
            return address
        time.sleep(delay)
    return ""


if __name__ == "__main__":
    eth_devices = EthDevices()
    ipv4_address = wait_for_ipv4_address(eth_devices.pcie)
    if not ipv4_address:
        print("ERROR: Could not determine the host's IPv4 address")
        exit(1)
    ipv6_address = eth_devices.usb_peer_ipv6()
    if not ipv6_address:
        print("ERROR: Could not discover the card's IPv6 address")
        exit(1)
    result = subprocess.run(
        [
            "socat",
            "-d",
            f"TCP4-LISTEN:2222,bind={ipv4_address},fork",
            f"TCP6:[{ipv6_address}%{eth_devices.usb}]:22",
        ]
    )
