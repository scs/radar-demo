#!/usr/bin/env python3
import subprocess


def get_enx_interface_uuid():
    # Execute nmcli to list connections and grep for 'enx' prefix
    try:
        output = subprocess.check_output(["nmcli", "device"], text=True)
        output = subprocess.check_output(
            ["nmcli", "-t", "-f", "NAME,UUID,DEVICE", "connection", "show"], text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute nmcli: {e}")
        return None, None, None

    # Search for the first 'enx' interface in the output
    for line in output.splitlines():
        name, uuid, device = line.split(":")
        if device.startswith("enx"):
            return name, uuid, device
    return None, None, None


def generate_yaml_file(name: str, uuid: str, interface_name: str):
    yaml_content = f"""
[connection]
id={name}
uuid={uuid}
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
    # Write the YAML content to a file
    with open(f"/etc/NetworkManager/system-connections/{name}.yaml", "w") as file:
        _ = file.write(yaml_content.strip())


def main():
    name, uuid, interface_name = get_enx_interface_uuid()
    if name and uuid and interface_name:
        generate_yaml_file(name, uuid, interface_name)
        print("Generated network-config.yaml successfully.")
    else:
        print("No 'enx' interface found.")


if __name__ == "__main__":
    main()
