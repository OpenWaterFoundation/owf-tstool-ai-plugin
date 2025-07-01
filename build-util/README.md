# build-util #

This folder contains scripts used in the build process.

| **Script** | **Description** |
| -- | -- |
| `0-create-plugin-jar.bash` | Create the plugin jar file in the user's `~/.tstool/NN/plugins/owf-tstool-timesheetscom-plugin` folder, used during development and before packaging the plugin for distribution. |
| `1-create-installer.bash` | Create the product installer in the repository `dist` folder. |
| `2-copy-to-owf-s3.bash` | Create a zip file for installation and copy to the OWF software.openwaterfoundation.org S3 bucket for public access. |
| `3-create-s3-index.bash` | Create the product landing page on software.openwaterfoundation.org. |
