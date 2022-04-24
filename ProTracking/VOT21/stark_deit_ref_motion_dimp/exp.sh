PROJECT_DIR=$(cd ../..; pwd)
sed -i "s@PROJECT_DIR@${PROJECT_DIR}@" trackers.ini
vot evaluate --workspace . iiau_rgbd
vot analysis --workspace . iiau_rgbd  --format html
vot pack --workspace . iiau_rgbd  # for submission
sed -i "s@${PROJECT_DIR}@PROJECT_DIR@" trackers.ini

echo " If exception is raised, try to re-activate the environment "
