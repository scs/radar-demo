#!/usr/bin/env python3
import re
import time

import serial


class FpgaSerial:

    def __init__(self):
        self.serial = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

    def flush_read(self):
        while True:
            time.sleep(0.1)
            response = self.serial.read_all()
            if not response:
                return

    def login(self):
        pattern = re.compile(".*-sh: root: command not found.*")
        while True:
            self.serial.write(b"root\n")
            time.sleep(0.1)
            binary_response = self.serial.read_all()

            if binary_response:
                responses = binary_response.decode("utf-8")
                for response in responses.splitlines():
                    if pattern.match(response):
                        return

    def logout(self):
        self.serial.write(b"exit\n")
        self.flush_read()

    def read_ip_v6(self) -> str | None:
        self.serial.write(b"ip -6 addr show eth0\n")
        # Wait for the command to execute and response to be returned
        time.sleep(0.1)
        # Read the response
        response = self.serial.read_all()
        if response:
            ipv6 = response.decode("utf-8").splitlines()[2].split()[1].split("/")[0]
            return ipv6

    def get_ipv6(self) -> str | None:
        self.login()
        self.flush_read()
        ipv6 = self.read_ip_v6()
        self.logout()
        return ipv6

    def modify_authorized_keys(self, public_key: str):
        public_key = public_key.strip()
        command = f'echo "{public_key}" >> ~/.ssh/authorized_keys\n'.encode("utf-8")
        print(command.decode("utf-8"))
        self.serial.write(command)
        self.flush_read()

    def append_authorized_key(self, public_key: str):
        print("appending to authorized_keys")
        self.login()
        self.flush_read()
        self.flush_read()
        self.flush_read()
        self.modify_authorized_keys(public_key)

    def close(self):
        self.serial.close()
