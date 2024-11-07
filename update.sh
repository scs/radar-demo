#!/bin/bash

git pull
pushd ansible || exit 1
ansible-playbook playbook.yaml -i localhost.yaml -K
popd || exit 1
ansible/roles/socat/files/update_fw.py -s test/

