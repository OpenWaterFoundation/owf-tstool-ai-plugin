#!/bin/bash
#
# Create the plugin jar file for installation in the deployed system
# - the class files and manifest are jar'ed up
# - the resulting Jar file is created in the user's (developer's)
#   folder consistent with the TSTool plugins folder:
#
#     TSTool 15+:
#       ./tstool/NN/plugins/owf-tstool-ai-plugin/1.0.0/
#
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

# Copy the maven dependencies into the plugin's 'dep' folder:
# - remove the 'dep' folder contents before copy to avoid version conflicts
copyMavenDependencies() {
  # First delete the 'dep' folder contents, but leave the folder itself.
  if [ -d "${pluginDepFolder}" ]; then
    # Remove all the existing files in the 'dep' folder so only the latest are available at runtime:
    # - only attempt to remove if not empty - otherwise get a warning
    # - do not use double quotes around the folder with wildcard
    if [ ! -z "$(ls -A ${pluginDepFolder})" ]; then
      rm ${pluginDepFolder}/*
    fi
  else
    # Dependency folder 'dep' does not exist.  Create it.
    mkdir -p "${pluginDepFolder}"
  fi

  # Copy the dependency jar files based on Maven listing:
  # - echoing a variable in double quotes causes the backslashes to be removed so convert to / before passing to linux commands
  echoStderr "[INFO] Copying jar files for dependencies to:"
  echoStderr "[INFO]   ${pluginDepFolder}"
  #mvn -f "${mavenPomFile}" dependency:build-classpath | grep ':' | grep ';' | tr ';' '\n' | tr '\\' '/'
  #return 0
  mvn -f "${mavenPomFile}" dependency:build-classpath | grep ':' | grep ';' | tr ';' '\n' | tr '\\' '/' | while read line
    # The Maven output on Windows from the above is similar to:
    #   C:\Users\sam\.m2\repository\io\netty\netty-common\4.1.77.Final\netty-common-4.1.77.Final.jar
    do
      # Convert to a path consistent with the Linux environment.
      #echo "Line from mvn: ${line}"
      linuxPath=$(cygpath -u "${line}")
      # For debugging.
      #echo "Linux path: ${linuxPath}"
      cp -v "${linuxPath}" "${pluginDepFolder}"
    done
}

# Echo a string to standard error (stderr).
# This is done so that TSTool results output printed to stdout is not mixed with stderr.
# For example, TSTool may be run headless on a server to output to CGI,
# where stdout formatting is important.
echoStderr() {
  ${echo2} "$@" >&2
}

# Get the plugin version (e.g., 1.2.0)
# - the version is printed to stdout so assign function output to a variable
getPluginVersion() {
  # Maven folder structure results in duplicate 'owf-tstool-ai-plugin'?
  # TODO smalers 2022-05-19 need to enable this.
  srcFile="${repoFolder}/owf-tstool-ai-plugin/src/main/java/org/openwaterfoundation/tstool/plugin/ai/PluginMeta.java"  
  # Get the version from the code line like:
  #   public static final String VERSION = "1.0.0 (2022-05-27)";
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
  local java8InstallHome java11InstallHome

  # Use the version of Java that is used for TSTool development.
  java8InstallHome='/C/Program Files/Java/jdk8'
  java11InstallHome='/C/Program Files/Java/jdk11'
  if [ -d "${java11InstallHome}" ]; then
    javaInstallHome="${java11InstallHome}"
    echoStderr "[INFO] Using Java 11: ${javaInstallHome}"
  elif [ -d "${java8InstallHome}" ]; then
    javaInstallHome="${java8InstallHome}"
    echoStderr "[INFO] Using Java 8: ${javaInstallHome}"
  else
    echoStderr ""
    echoStderr "${errorColor}[ERROR] Unable to determine Java location.  Exiting,${endColor}"
    exit 1
  fi

  # Also set JAVA_HOME, needed by Maven.
  export JAVA_HOME="${javaInstallHome}"
}

# Main entry point.

# Configure the echo command to output color:
# - do this first because it is used for logging messages
configureEcho

# Make sure the operating system is supported.
checkOperatingSystem

# Get the location where this script is located since it may have been run from any folder.
scriptFolder=$(cd $(dirname "$0") && pwd)
repoFolder=$(dirname ${scriptFolder})
gitReposFolder=$(dirname ${repoFolder})
tstoolMainRepoFolder=${gitReposFolder}/cdss-app-tstool-main

mavenProjectFolder=${repoFolder}/owf-tstool-ai-plugin
mavenPomFile=${mavenProjectFolder}/pom.xml

# Get the plugin version, which is used in the jar file name.
pluginVersion=$(getPluginVersion)
if [ -z "${pluginVersion}" ]; then
  echoStderr "${errorColor}[ERROR] Unable to determine plugin version.${endColor}"
  exit 1
else
  echoStderr "[INFO] Plugin version:  ${pluginVersion}"
fi

# TODO smalers 2019-06-16 figure out how to handle different TSTool/plugin versions.
#tstoolVersion=12
#tstoolVersion=13
tstoolMajorVersion=$(getTSToolMajorVersion)
if [ -z "${tstoolMajorVersion}" ]; then
  echoStderr "${errorColor}[ERROR] Unable to determine TSTool main version.${endColor}"
  exit 1
else
  echoStderr "[INFO] TSTool main version:  ${tstoolMajorVersion}"
fi

# Standard locations for plugin files:
# - put after determining versions
# - the folders adhere to Maven folder structure
devBinFolder="${repoFolder}/owf-tstool-ai-plugin/target/classes"

# Main folder for installed plugins.
pluginsFolder="$HOME/.tstool/${tstoolMajorVersion}/plugins"

# Main installed folder for the plugin.
mainPluginFolder="${pluginsFolder}/owf-tstool-ai-plugin"

# Version installed folder for the plugin.
versionPluginFolder="${mainPluginFolder}/${pluginVersion}"

jarFile="${versionPluginFolder}/owf-tstool-ai-plugin-${pluginVersion}.jar"
manifestFile="${repoFolder}/owf-tstool-ai-plugin/src/main/resources/META-INF/MANIFEST.MF"

# Folder for plugin version dependencies.
pluginDepFolder="${versionPluginFolder}/dep"

# Set the javaInstallHome variable.
setJavaInstallHome

# Create the jar file in user's development files.
echoStderr "[INFO] Creating a jar file from class files in folder:  ${devBinFolder}"
echoStderr "[INFO] Manifest file for jar file:  ${manifestFile}"
echoStderr "[INFO] Jar file: ${jarFile}"

# Remove the jar file first to make sure it does not append.
#rm ${jarFile}
if [ ! -d "${devBinFolder}" ]; then
  echoStderr ""
  echoStderr "${errorColor}[ERROR] Project bin folder does not exist:  ${devBinFolder}${endColor}"
  echoStderr "${errorColor}[ERROR] Make sure to compile software in Eclipse.${endColor}"
  exit 1
fi

# Make sure the TSTool plugins folder exists.
if [ ! -d "${pluginsFolder}" ]; then
  echoStderr ""
  echoStderr "[INFO] TSTool plugins folder does not exist:  ${pluginsFolder}"
  echoStderr "[INFO] Creating it."
  mkdir "${pluginsFolder}"
fi

# Make sure the main folder exists for the plugin.
if [ ! -d "${mainPluginFolder}" ]; then
  echoStderr ""
  echoStderr "[INFO] Plugin main folder does not exist:  ${mainPluginFolder}"
  echoStderr "[INFO] Creating it."
  mkdir "${mainPluginFolder}"
fi

# Make sure the version folder exists for the plugin.
if [ ! -d "${versionPluginFolder}" ]; then
  echoStderr ""
  echoStderr "[INFO] Plugin version folder does not exist:  ${versionPluginFolder}"
  echoStderr "[INFO] Creating it."
  mkdir "${versionPluginFolder}"
fi

cd ${devBinFolder}
"${javaInstallHome}/bin/jar" -cvfm ${jarFile} ${manifestFile} *
if [ ! "$?" = "0" ]; then
  echoStderr "${errorColor}[ERROR] Error creating jar file.  Exiting.${endColor}"
  exit 1
fi

# Echo out the jar file contents.
echoStderr "[INFO] Listing of jar file that was created..."
"${javaInstallHome}/bin/jar" -tvf ${jarFile}

# Print the java file location again and check for duplicate jar files.
echoStderr ""
echoStderr "[INFO] Jar file is:  ${jarFile}"
jarCount=$(ls -1 ${versionPluginFolder} | grep -v 'dep' | grep -v 'zip' | wc -l)
if [ ${jarCount} -eq 1 ]; then
  echoStderr "[INFO] 1 plugin jar file is installed (see below).  OK."
  # Do not put quotes around the following.
  ls -1 ${versionPluginFolder}/*.jar
else
  echoStderr "${errorColor}[ERROR] ${jarCount} plugin jar files are installed (see below).${endColor}"
  echoStderr "${errorColor}[ERROR] There should only be one, typically the latest version.${endColor}"
  echoStderr "${errorColor}[ERROR] Remove old versions or move to 'plugins-old' in case need to restore.${endColor}"
  # Do not put quotes around the following.
  ls -1 ${versionPluginFolder}/*.jar
  exit 1
fi

# Copy jar file dependencies from Maven to the plugin 'dep' folder.
copyMavenDependencies

exit 0
