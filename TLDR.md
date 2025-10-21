# Short instructions for updating the system
1. Pull the latest changes from the repository.

   ```bash
   git pull
   ```

2. Switch to changes_for_3d_visuals branch

   ```bash
   > git switch changes_for_3d_visuals
   ```

3. Update the system

   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

4. Run ansible script

   ```bash
    > cd ~/radar_demo/ansible 
    > ansible-playbook playbook.yaml -i localhost.yaml -K
   ```

4. Update FPGA bitstream

   ```bash
   > cd ~/radar_demo/update_fw/
   > python3 update_fw.py -s hw
   ```

5. Connect to the FPGA board with tio

    ```bash
    > tio /dev/ttyUSB0
    ```
6. Check that the files have been written to the board and delete the Image.ub file if it is present

    ```bash
    root@fpga-board:~# ls 
    BOOT.BIN boot.scr hw.xclbin Image ssh-user-image-radar_demo.cpio.gz.u-boot system.dtb
    root@fpga-board:~# rm Image.ub
    ```
7. Optional check the MD5 checksum of the files

8. Reboot the FPGA board and interrupt the boot process to access the U-Boot console

    ```bash
    root@fpga-board:~# reboot
    Hit any key to stop autoboot:  5

    ```

9. Configure U-Boot according to the used firmware version below.
    a. Start from the default environment

    ```bash
    radar> env default -f -a
    ```

    b. Set the boot device

    ```bash
    radar> setenv boot_targets mmc1 mmc0 jtag
    ```

    c. Set the mac address to the address on the **sticker** on the card

    ```bash
    radar> setenv ethaddr a0:a6:5c:00:09:cc
    ```

    d. Set the hostname with extrabootargs

    ```bash
    radar> setenv extrabootargs systemd.hostname=radardemo-<yourchoice>
    ```

    e. Set the boot script address (to be fixed in yocto)

    ```bash
    radar> setenv scriptaddr 0x20000000
    ```

    f. Save the settings

    ```bash
    radar> saveenv
    ```

    g. Boot the card

    ```bash
    radar> boot
    ```

10. Cold boot PC (power cycle)




# Switching to non 3D visual version

1. Switch to branch main_new_uboot
2. Delete files in ~/radar_demo/backend/stimuli/radardemo_dcr_4tx16rx*.bin
