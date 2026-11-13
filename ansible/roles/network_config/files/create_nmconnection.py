#!/usr/bin/env python3
import os
import subprocess
import time
import uuid


def find_enx_device():
    # Look at known devices directly instead of relying on NetworkManager
    # having already created/bound a connection profile to it.
    try:
        output = subprocess.check_output(
            ["nmcli", "-t", "-f", "DEVICE,TYPE", "device", "status"], text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute nmcli: {e}")
        return None

    for line in output.splitlines():
        device = line.split(":")[0]
        if device.startswith("enx"):
            return device
    return None


def generate_yaml_file(name: str, connection_uuid: str, interface_name: str):
    yaml_content = f"""
[connection]
id={name}
uuid={connection_uuid}
type=ethernet
autoconnect-priority=-999
interface-name={interface_name}

[ethernet]

[ipv4]
method=disabled

[ipv6]
addr-gen-mode=stable-privacy
method=link-local

[proxy]

"""
    path = f"/etc/NetworkManager/system-connections/{name}.nmconnection"
    with open(path, "w") as file:
        _ = file.write(yaml_content.strip())
    # NetworkManager ignores connection files that are group/world readable.
    os.chmod(path, 0o600)


def main():
    device = None
    for _ in range(10):
        device = find_enx_device()
        if device:
            break
        time.sleep(1)

    if not device:
        print("No 'enx' interface found.")
        exit(1)

    generate_yaml_file("usb-ethernet", str(uuid.uuid4()), device)
    print("Generated usb-ethernet.nmconnection successfully.")
    subprocess.run(["nmcli", "connection", "reload"], check=False)


if __name__ == "__main__":
    main()
