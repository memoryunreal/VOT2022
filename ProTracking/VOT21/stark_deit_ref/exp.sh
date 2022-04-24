PROJECT_DIR=$(cd ../..; pwd)
sed -i "s@PROJECT_DIR@${PROJECT_DIR}@" trackers.ini
vot evaluate --workspace . stark_deit_ref
vot analysis --workspace . stark_deit_ref  --format html
sed -i "s@${PROJECT_DIR}@PROJECT_DIR@" trackers.ini

