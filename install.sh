export PROJ_HOME=$(pwd)

# Chamfer Distance & emd
cd $PROJ_HOME/extensions/chamfer_dist
python setup.py install --user
# cd $PROJ_HOME/extensions/emd
# python setup.py install --user