echo "5X5"
python maze/create_expert.py --path ./maze/environment/mazes/mazes5 --width 5 --height 5 --save_path ./dataset/dataset5

echo "10X10"
python maze/create_expert.py --path ./maze/environment/mazes/mazes10 --width 10 --height 10 --save_path ./dataset/dataset10

echo "25X25"
python maze/create_expert.py --path ./maze/environment/mazes/mazes25 --width 25 --height 25 --save_path ./dataset/dataset25

echo "50X50"
python maze/create_expert.py --path ./maze/environment/mazes/mazes50 --width 50 --height 50 --save_path ./dataset/dataset50

echo "75X75"
python maze/create_expert.py --path ./maze/environment/mazes/mazes75 --width 75 --height 75 --save_path ./dataset/dataset75

echo "100X100"
python maze/create_expert.py --path ./maze/environment/mazes/mazes100 --width 100 --height 100 --save_path ./dataset/dataset100