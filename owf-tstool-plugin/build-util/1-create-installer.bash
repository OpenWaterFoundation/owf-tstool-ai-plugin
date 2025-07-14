#!/bin/bash
#
# Create the plugin installer file for the download website:
# - the current files in the following folder are zipped, keeping the plugin main folder:
#   ./tstool/NN/plugins/owf-tstool-timesheetscom-plugin/${pluginVersion}
# - output is to the plugin 'dist' folder with filename like:
#     tstool-timesheetscom-plugin-1.0.0-win-2206060102.zip
#   where the number is YYMMDDHHMM

# Supporting functions, alphabetized.

# Determine the operating system that is running the script:
# - mainly care whether Cygwin
checkOperatingSystem() {
  if [ ! -z "${operatingSystem}" ]; then
    # Have already checked operating system so return.
    return
  fi
  operatingSystem="unknown"
  os=$(uname | tr [a-z] [A-Z])
  case "${os}" in
    CYGWIN*)
      operatingSystem="cygwin"
      ;;
    LINUX*)
      operatingSystem="linux"
      ;;
    MINGW*)
      operatingSystem="mingw"
      ;;
  esac
  echoStderr "[INFO] operatingSystem=${operatingSystem} (used to check for Cygwin and filemode compatibility)"

  if [ "${operatingSystem}" != "mingw" ]; then
    echoStderr "${errorColor}[ERROR] Currently this script only works for MINGW (Git Bash)${endColor}"
    exit 1
  fi
}

# Determine which echo to use, needs to support -e to output colored text:
# - normally built-in shell echo is OK, but on Debian Linux dash is used, and it does not support -e
configureEcho() {
  echo2='echo -e'
  testEcho=$(echo -e test)
  if [ "${testEcho}" = '-e test' ]; then
    # The -e option did not work as intended:
    # - using the normal /bin/echo should work
    # - printf is also an option
    echo2='/bin/echo -e'
    # The following does not seem to work.
    #echo2='printf'
  fi

  # Strings to change colors on output, to make it easier to indicate when actions are needed:
  # - Colors in Git Bash:  https://stackoverflow.com/questions/21243172/how-to-change-rgb-colors-in-git-bash-for-windows
  # - Useful info:  http://webhome.csc.uvic.ca/~sae/seng265/fall04/tips/s265s047-tips/bash-using-colors.html
  # - See colors:  https://en.wikipedia.org/wiki/ANSI_escape_code#Unix-like_systems
  # - Set the background to black to eensure that white background window will clearly show colors contrasting on black.
  # - Yellow "33" in Linux can show as brown, see:  https://unix.stackexchange.com/questions/192660/yellow-appears-as-brown-in-konsole
  # - Tried to use RGB but could not get it to work - for now live with "yellow" as it is
  warnColor='\e[1;40;93m' # user needs to do something, 40=background black, 33=yellow, 93=bright yellow
  errorColor='\e[0;40;31m' # serious issue, 40=background black, 31=red
  menuColor='\e[1;40;36m' # menu highlight 40=background black, 36=light cyan
  okColor='\e[1;40;32m' # status is good, 40=background black, 32=green
  endColor='\e[0m' # To switch back to default color
}

# Zip the plugin files.
createPluginZipFile() {
  # Make sure that the 7zip program exists.
  if [ ! -f "${sevenzip}" ]; then
    echoStderr "[ERROR] 7zip software does not exist:  ${sevenzip}"
    exit 1
  fi
  # Remove the old zip file to make sure there is no confusion creating the file.
  if [ -f "${zipFile}" ]; then
    echoStderr "[INFO] Removing existing zip file:  ${zipFile}"
    rm -f "${zipFile}"
  fi
  # Change to the plugins folder:
  # - only want to include the specific version,
  #   but also include the parent of the version folder
  cd ${pluginsFolder}
  echoStderr "[INFO] Running 7zip to create zip file:  ${zipFile}"
  echoStderr "[INFO] Current folder is:  ${pluginsFolder}"
  "${sevenzip}" a -tzip ${zipFile} "owf-tstool-timesheetscom-plugin/${pluginVersion}"
  exitStatus=$?
  if [ ${exitStatus} -ne 0 ]; then
    echoStderr "[INFO] Error running 7zip, exit status (${exitStatus})."
    exit 1
  fi
}

# Echo a string to standard error (stderr).
# This is done so that TSTool results output printed to stdout is not mixed with stderr.
# For example, TSTool may be run headless on a server to output to CGI,
# where stdout formatting is important.
echoStderr() {
  ${echo2} "$@" 1>&2
}

# Get the plugin version (e.g., 1.2.0):
# - the version is printed to stdout so assign function output to a variable
getPluginVersion() {
  local srcFile

  # Maven folder structure results in duplicate 'owf-tstool-timesheetscom-plugin'?
  # TODO smalers 2022-05-19 need to enable this.
  srcFile="${repoFolder}/owf-tstool-timesheetscom-plugin/src/main/java/org/openwaterfoundation/tstool/plugin/timesheetscom/PluginMeta.java"  
  # Get the version from the code line like:
  #  public static final String VERSION = "1.0.0 (2022-05-27)";
  if [ -f "${srcFile}" ]; then
    cat ${srcFile} | grep 'VERSION =' | cut -d '"' -f 2 | cut -d ' ' -f 1 | tr -d '"' | tr -d ' '
  else
    # Don't echo error to stdout.
    echoStderr "[ERROR] Source file with version does not exist:"
    echoStderr "[ERROR]   ${srcFile}"
    # Output an empty string as the version.
    echo ""
  fi
}

# Get the TSTool major version (e.g., "13" for 13.3.0):
# - the version is printed to stdout so assign function output to a variable
getTSToolMajorVersion() {
  srcFile="${tstoolMainRepoFolder}/src/DWR/DMI/tstool/TSToolMain.java"  
  # Get the version from the code line like:
  #   this.pluginProperties.put("Version", "1.2.0 (2020-05-29");
  cat ${srcFile} | grep 'public static final String PROGRAM_VERSION' | cut -d '=' -f 2 | cut -d '(' -f 1 | tr -d ' ' | tr -d '"' | cut -d '.' -f 1
}

# Determine the Java install home, consistent with TSTool development environment.
setJavaInstallHome() {
  javaInstallHome='/C/Program Files/Java/jdk8'
  if [ ! -d "${javaInstallHome}" ]; then
    echoStderr ""
    echoStderr "${errorColor}[ERROR] Unable to determine Java location.  Exiting,${endColor}"
    exit 1
  fi
}

# Main entry point.

# Configure the echo command to output color.
configureEcho

# Make sure the operating system is supported.
checkOperatingSystem

# Get the location where this script is located since it may have been run from any folder.
scriptFolder=$(cd $(dirname "$0") && pwd)
repoFolder=$(dirname ${scriptFolder})
distFolder="${repoFolder}/dist"
gitReposFolder=$(dirname ${repoFolder})
tstoolMainRepoFolder="${gitReposFolder}/cdss-app-tstool-main"

# Zip up all the plugin files:
# - use 7zip command line (must be installed in normal location)
# - zip with the folder so that there is less chance to clobber existing files
# - use a list file to control what files are included
# - the zip file includes the StateMod version from statem.for file
sevenzip='/C/Program Files/7-Zip/7z.exe'

# Get the TSTool major version to find the installed files.
tstoolMajorVersion=$(getTSToolMajorVersion)
if [ -z "${tstoolMajorVersion}" ]; then
  echoStderr "${errorColor}[ERROR] Unable to determine TSTool major version.${endColor}"
  exit 1
else
  echoStderr "[INFO] TSTool major version:  ${tstoolMajorVersion}"
fi

# Get the plugin version, which is used in the installer file name.
pluginVersion=$(getPluginVersion)
if [ -z "${pluginVersion}" ]; then
  echoStderr "${errorColor}[ERROR] Unable to determine plugin version.${endColor}"
  exit 1
else
  echoStderr "[INFO] Plugin version:  ${pluginVersion}"
fi

# Standard locations for plugin files:
# - put after determining versions
# - the folders adhere to Maven folder structure
devBinFolder="${repoFolder}/owf-tstool-timesheetscom-plugin/target/classes"

# Main folder for installed plugins.
pluginsFolder="${HOME}/.tstool/${tstoolMajorVersion}/plugins"

# Main installed folder for the plugin.
mainPluginFolder="${pluginsFolder}/owf-tstool-timesheetscom-plugin"

# Version installed folder for the plugin.
versionPluginFolder="${mainPluginFolder}/${pluginVersion}"

# Jar file for the plugin.
jarFile="${versionPluginFolder}/owf-tstool-timesheetscom-plugin-${pluginVersion}.jar"

# Plugin dependency folder.
pluginDepFolder="${versionPluginFolder}/dep"

now=$(date +%Y%m%d%H%M)
zipFile="${distFolder}/tstool-timesheetscom-plugin-${pluginVersion}-win-${now}.zip"

# Create the local plugin files to make sure they are current.
echoStderr "[INFO] Creating the jar file with current development files..."
${scriptFolder}/0-create-plugin-jar.bash
if [ $? -ne 0 ]; then
  echoStderr "[ERROR] Error creating plugin jar file."
  exit 1
fi

# Zip up the plugin files and create a zip file in the 'dist' folder.
createPluginZipFile

echoStderr "[INFO] Successfully created the zip file:"
echoStderr "[INFO]   ${zipFile}"

exit 0
