#!/bin/bash

SHOTDETECTORBIN=/home/chuck/projects/Shotdetect/build/shotdetect-cmd
VIDEODIR=/media/6TB/Videos/dextro-benchmark-2016-03-30

for f in ${VIDEODIR}/*.{mp4,mov}; do
  NAME=${f%.*}
  BNAME=`basename ${NAME}`_shot
  echo Processing ${BNAME}
  mkdir -p ${VIDEODIR}/${BNAME} >& /dev/null
  echo Running: ${SHOTDETECTORBIN} -i ${f} -o ${VIDEODIR}/${BNAME} -s 60 -v -f -a `basename ${NAME}`
  ${SHOTDETECTORBIN} -i ${f} -o ${VIDEODIR}/${BNAME} -s 60 -v -f -a `basename ${NAME}`
done

echo -----------------------------------------------------------
echo Done!
