#!/bin/bash

# two modifications to the original script UCB provided
# 1: don't use FPS
# 2: "%6d" format for image numbering (no more "%4d")

#EXPECTED_ARGS=2
# fps won't be used here
EXPECTED_ARGS=1
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  #echo "Usage: `basename $0` video frames/sec [size=256]"
  echo "Usage: `basename $0` video [size=256]"
  exit $E_BADARGS
fi

NAME=${1%.*}
#FRAMES=$2
BNAME=`basename $NAME`
echo $BNAME
mkdir -m 755 $BNAME

#ffmpeg -i $1 -r $FRAMES $BNAME/$BNAME.%4d.jpg
ffmpeg -i $1 $BNAME/$BNAME.%6d.jpg
