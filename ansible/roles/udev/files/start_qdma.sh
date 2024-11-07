#!/bin/bash

for DEVICE_ID in b024 b028 b034 b038 b044 b048; do
  FOUND=$(lspci -nD | grep -c "${VENDOR_ID}":"${DEVICE_ID}")
  if [ "${FOUND}" -eq 1 ]; then

    SLOT_NR=$(lspci -nD | grep -i ${VENDOR_ID}:${DEVICE_ID} | cut -d " " -f 1)
    break
  fi
done

if [ -z ${SLOT_NR+x} ]; then
  echo "PCIE for vendor ${VENDOR_ID} card not found"
  exit 0
else
  echo "PCIE for vendor ${VENDOR_ID} card found with device id ${DEVICE_ID} and slot nr ${SLOT_NR}"
fi

DOMAIN=$(echo "${SLOT_NR}" | cut -d ":" -f 1)
BUS=$(echo "${SLOT_NR}" | cut -d ":" -f 2)
DEVICE_AND_FUNCTION=$(echo "${SLOT_NR}" | cut -d ":" -f 3)
DEVICE=$(echo "${DEVICE_AND_FUNCTION}" | cut -d "." -f 1)
FUNCTION=$(echo "${SLOT_NR}" | cut -d "." -f 2)

DEV=qdma${BUS}${DEVICE}${FUNCTION}

echo 512 > "/sys/bus/pci/devices/${SLOT_NR}/qdma/qmax"
/usr/local/sbin/dma-ctl "${DEV}" q add idx 0 mode mm dir h2c
/usr/local/sbin/dma-ctl "${DEV}" q start idx 0 dir h2c
/usr/local/sbin/dma-ctl "${DEV}" q add idx 1 mode mm dir c2h
/usr/local/sbin/dma-ctl "${DEV}" q start idx 1 dir c2h
chown root:users "/dev/${DEV}-MM-0"
chown root:users "/dev/${DEV}-MM-1"
chmod 0660 "/dev/${DEV}-MM-0"
chmod 0660 "/dev/${DEV}-MM-1"
chown root:users "/dev/${DEV}-MM-0"

chown root:users "/sys/bus/pci/devices/${SLOT_NR}/resource2"
chmod 0660 "/sys/bus/pci/devices/${SLOT_NR}/resource2"
