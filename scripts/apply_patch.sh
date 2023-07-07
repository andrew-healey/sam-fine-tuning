echo "Patch: $1"
cd segment-anything/
git apply ../patches/$1
echo "Done!"