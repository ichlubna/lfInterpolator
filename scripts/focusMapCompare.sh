POSITIONS=(0.071 0.193714 0.316429 0.439143 0.561857 0.684571 0.807286 0.93)
SCENES=(lowFrequency lowDepth bonfire cornell simpleSetting)
FOCUS_STARTS=(0.0 0.54 0.06 0.22 0.43)
FOCUS_ENDS=(0.46 0.09 0.24 0.17 0.18)
ASPECTS=(2.0223 2.122 2.276 1.783 1.8266)

TEMP=$(mktemp -d)
#The offsets need to be edited in the interpolator according to the view id
VIEW=0
for ((i = 0 ; i < 5 ; i++)); do
    ../build/lfInterpolator -i ../../lfStreaming/scripts/finalResult/inputs/${SCENES[$i]} -o $TEMP/ -t 0.071,0.071,0.93,0.93 -f ${FOCUS_STARTS[$i]} -r ${FOCUS_ENDS[$i]} -m STD -s 7 -a ${ASPECTS[$i]}
    LEAD=$(printf "%02d" $VIEW)
    OUT_COMMON=./comparison/${SCENES[$i]}C
    OUT=./comparison/${SCENES[$i]}
    mkdir -p $OUT_COMMON
    mkdir -p $OUT
    mv $TEMP/$LEAD.png $OUT_COMMON/$VIEW.png 
    POS=${POSITIONS[$VIEW]}
    ../build/lfInterpolator -i ../../lfStreaming/scripts/finalResult/inputs/${SCENES[$i]} -o $TEMP/ -t $POS,$POS,$POS,$POS -f ${FOCUS_STARTS[$i]} -r ${FOCUS_ENDS[$i]} -m STD -s 7 -a ${ASPECTS[$i]}
    mv $TEMP/00.png $OUT/$VIEW.png 
done
rm -rf $TEMP
