#!/usr/bin/env python3
import subprocess


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
