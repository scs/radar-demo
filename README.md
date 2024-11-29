# radar-demo
Artefacts for radar-demo

## Installation

#### USB Ethernet cable
Download driver from here [ASIX Driver](https://www.asix.com.tw/en/support/download/file/1892?time=1732778829482)

```
> cd Downloads
> tar -jxvf ASIX_USB_NIC_Linux_Driver_Source_v3.4.0.tar.bz2
> cd ASIX_USB_NIC_Linux_Driver_Source_v3.4.0
> make
> sudo make install
```

Instructions from here [Reddit](https://www.reddit.com/r/HomeNetworking/comments/1ftjlrt/ethernet_over_usb_is_not_working_in_ubuntu_22/?rdt=40984)


### System
1. git clone git@github.com:scs/radar-demo
2. run radar-demo/update.sh

This will update the repo, install all relevant dependencies and update the
card. It is possible that a reboot is required after running the update script.
Please do not stop the update once started.

For the update the password of the current user is required to gain root
privilidges.

### Card Update
__** Impportant: **__
This can only be done once all previous steps have been completed and the PC has been rebooted.

```console
> cd update_fw
> ./update_fw.py -s hw

```
If propmted please input the password of the user


## Running the application
After installation a launcer is installed. One can call it initially by
starting the desktop drawer. This is done with the meta (Windows) key. Then
typing radar-demo should show an radar icon. This can be clicked or confirmed
with enter. To keep the icon on the launcher bar, right click and click 'Add to
Favorites'.

If when starting the application the browser does not find the page, be patient
and reload the page.

