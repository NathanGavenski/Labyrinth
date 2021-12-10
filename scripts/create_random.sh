echo "5X5"
python maze/create_random.py --path ./maze/environment/mazes/mazes5 --width 5 --height 5 --save_path ./dataset/random_dataset5 --amount 10000

# echo "10X10"
# xvfb-run  -s "-screen 0 1400x900x24" python maze/create_random.py --path ./maze/environment/mazes/mazes10 --width 10 --height 10 --save_path ./dataset/random_dataset10 --amount 10000

# echo "25X25"
# xvfb-run  -s "-screen 0 1400x900x24" python maze/create_random.py --path ./maze/environment/mazes/mazes25 --width 25 --height 25 --save_path ./dataset/random_dataset25 --amount 10000

# echo "50X50"
# xvfb-run  -s "-screen 0 1400x900x24" python maze/create_random.py --path ./maze/environment/mazes/mazes50 --width 50 --height 50 --save_path ./dataset/random_dataset50 --amount 10000

# echo "75X75"
# xvfb-run  -s "-screen 0 1400x900x24" python maze/create_random.py --path ./maze/environment/mazes/mazes75 --width 75 --height 75 --save_path ./dataset/random_dataset75 --amount 10000

# echo "100X100"
# xvfb-run  -s "-screen 0 1400x900x24" python maze/create_random.py --path ./maze/environment/mazes/mazes100 --width 100 --height 100 --save_path ./dataset/random_dataset100 --amount 10000