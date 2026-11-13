#!/usr/bin/env python3
import re
import subprocess
import time


class EthDevices:
    def __init__(self):
        # Execute the "ip link" command
        result = subprocess.run(
            ["ip", "link"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Split the output into lines for processing
        lines = result.stdout.split("\n")

        self.devices: list[str] = []
        # Iterate over each line to find and print the device names
        for line in lines:
            if ": " in line:  # Check if the line contains a device name
                parts = line.split(": ")
                if len(parts) > 1:
                    _, details = parts[0], parts[1]
                    # Split off any '@' part for virtual interfaces
                    device_name = details.split("@")[0]
                    self.devices.append(device_name)

    @property
    def pcie(self) -> str:
        for device in self.devices:
            if device.startswith("enp"):
                return device
        return ""

    @property
    def usb(self) -> str:
        for device in self.devices:
            if device.startswith("enx"):
                return device
        return ""

    def usb_peer_ipv6(self, attempts: int = 10, delay: float = 1.0) -> str | None:
        """Discover the link-local IPv6 address of whatever is on the other
        end of the USB ethernet link, via plain IPv6 Neighbor Discovery.
        This is a point-to-point link, so there can only be one neighbor."""
        device = self.usb
        if not device:
            return None

        pattern = re.compile(
            r"^(?P<addr>[a-fA-F0-9:]+)\s+lladdr\s+\S+\s+(?P<state>\S+)"
        )
        for _ in range(attempts):
            # Trigger Neighbor Discovery by pinging the all-nodes multicast
            # address; any reply populates the neighbor table below.
            subprocess.run(
                ["ping", "-6", "-c", "1", "-W", "1", "-I", device, "ff02::1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            result = subprocess.run(
                ["ip", "-6", "neighbor", "show", "dev", device],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in result.stdout.splitlines():
                if m := pattern.match(line):
                    if m.group("state") in ("REACHABLE", "STALE", "DELAY", "PERMANENT"):
                        return m.group("addr")
            time.sleep(delay)
        return None
