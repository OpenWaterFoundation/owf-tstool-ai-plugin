# Setting up the AI Plugin

This documentation explains how to initialize the TSTool AI plugin.
This is typically only done once during initial setup,
and then the repository can be added to the TSTool Eclipse workspace using the instructions in the
main repository [`README.md`](../README.md) file.

## Initialize the Repository ##

Create the GitHub public repository `owf-tstool-ai-plugin`.

Clone to the `TSTool/git-repos` folder on the computer.

Copy the `README.md`, `.gitignore`, `.gitattributes`, and `build-util/` files from the `owf-tstool-ai-plugin` repository.
The `README.md` and `build-util/` files will be updated as the code is updated.

## Add Maven Project to Eclipse ##

Start the TSTool Eclipse environment by running the `cdss-app-tstool-main/build-util/run-eclipse-win64.cmd` command file
from a Windows command shell.

Use the Eclipse ***File / New / Project... / Maven / Maven Project***.  Specify information as shown in the following image.
Redundant `owf-tstool-ai-plugin` folders are used, one for the Git repository folder working files,
and one for the Maven project with source files.
Create the new folder using the selector dialog if it does not exist.
This allows other top-level folders to be created in the repository to separate major development files, including documentation and tests.

![New Maven project, step 1](new-maven-project1.png)

Press ***Next >***.  Fill out the new maven project artifact properties as follows:

![New Maven project, step 2](new-maven-project2.png)

Press ***Finish***.

The project will be shown in Eclipse and folders and files will be initialized for the Maven project.

Next, copy files from other similar plugins to implement the functionality.
The `build-util` scripts can be used to create the plugin `jar` file and run the plugin during development.
