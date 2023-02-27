#!/bin/bash

for shape in 2 3 4 5 6 7 8 9 10
    do
        echo "Current shape: ${shape}" 
        python inputchanger.py --shape $shape
    

    done

