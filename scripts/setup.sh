echo "Creating mazes"
./scripts/create_maze.sh

echo "Creating expert datasets"
./scripts/create_expert.sh

echo "Creating random dataset"
./scripts/create_random.sh