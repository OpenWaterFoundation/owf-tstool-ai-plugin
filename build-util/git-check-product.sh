#!/bin/sh
(set -o igncr) 2>/dev/null && set -o igncr; # This comment is required.
# The above line ensures that the script can be run on Cygwin/Linux even with Windows CRNL.
#
# git-check-product [--onlyplugin] ...
# - specify --onlyplugin to NOT prompt to check TSTool main repositories
# - check the TSTool repositories for status
# - this script calls the general git utilities script

# Get the location where this script is located since it may have been run from any folder.
scriptFolder=$(cd $(dirname "$0") && pwd)

# The --onlyplugin parameter is used with the TSTool git-check-plugins script:
# - possible command lines may include the following, do simple command line parsing:
#    --onlyplugin
#    or --debug --onlyplugin
#
# -if --onlyplugin is specified, don't prompt to check the main TSTool repositories.
onlyplugin="false"
if [ $# -ge 1 -a "$1" = "--onlyplugin" ]; then
  onlyplugin="true"
elif [ $# -ge 2 -a "$2" = "--onlyplugin" ]; then
  onlyplugin="true"
fi

# Main repository.
mainRepo="owf-tstool-timesheetscom-plugin"

# Git utilities folder is relative to the user's files in a standard development files location:
# - determine based on location relative to the script folder
# Specific repository folder for this repository.
repoFolder=$(dirname ${scriptFolder})
gitReposFolder=$(dirname ${repoFolder})

# Run the general script.
${scriptFolder}/git-util/git-check.sh --mainrepo="${mainRepo}" --reposfolder="${gitReposFolder}" $@

# Also check the main TSTool product repositories.
if [ "${onlyplugin}" = "false" ]; then
  tstoolBuildUtilFolder=${gitReposFolder}/cdss-app-tstool-main/build-util
  echo ""
  read -p "Also check the main TSTool product (y/N)? " answer
  if [ "${answer}" = "y" -o "${answer}" = "Y" ]; then
    ${tstoolBuildUtilFolder}/git-check-tstool.sh
  fi
  echo ""
fi

exit 0
