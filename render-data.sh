


# for world_map in "Town01" "Town02" "Town03" "Town04" "Town05" "Town06" "Town07"; do
for world_map in "Town10HD"; do
    echo "Rendering $world_map"
    python render-data.py --world_map $world_map
done