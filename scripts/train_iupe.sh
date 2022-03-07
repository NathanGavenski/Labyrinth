echo "5x5"
xvfb-run --auto-servernum --server-num=1 python train_iupe.py --gpu 0 --size 5

echo "10x10"
xvfb-run --auto-servernum --server-num=1 python train_iupe.py --gpu 0 --size 10

echo "25x25"
xvfb-run --auto-servernum --server-num=1 python train_iupe.py --gpu 0 --size 25

echo "50x50"
xvfb-run --auto-ervernum --server-num=1 python train_iupe.py --gpu 0 --size 50

echo "75x75"
xvfb-run --auto-ervernum --server-num=1 python train_iupe.py --gpu 0 --size 75

echo "100x100"
xvfb-run --auto-ervernum --server-num=1 python train_iupe.py --gpu 0 --size 100
