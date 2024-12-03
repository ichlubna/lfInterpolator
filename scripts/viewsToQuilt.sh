#!/bin/bash
montage "$1/*.png" -tile 5x9 -geometry 1920x1080+0+0 "$1/quilt.jpg"

