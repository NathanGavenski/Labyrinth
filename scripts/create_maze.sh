echo "5X5"
xvfb-run -a -s "-screen 1 1400x900x24" python maze/generate.py --width 5 --height 5 --train 100 --eval 100

echo "10X10"
xvfb-run -a -s "-screen 1 1400x900x24" python maze/generate.py  --width 10 --height 10 --train 100 --eval 100

echo "25X25"
xvfb-run -a -s "-screen 1 1400x900x24" python maze/generate.py --width 25 --height 25 --train 100 --eval 100

echo "50X50"
xvfb-run -a -s "-screen 1 1400x900x24" python maze/generate.py  --width 50 --height 50 --train 100 --eval 100

echo "75X75"
xvfb-run -a -s "-screen 1 1400x900x24" python maze/generate.py --width 75 --height 75 --train 100 --eval 100

echo "100X100"
xvfb-run -a -s "-screen 1 1400x900x24" python maze/generate.py --width 100 --height 100 --train 100 --eval 100
