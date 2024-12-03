for file in $1/*; do
    base=$(basename $file) 
    ./imageQualityMetrics.sh $1/$base $2/$base
done
