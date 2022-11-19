CKPT_PATH="ckpt/martin_room_$1"
DATASET_PATH=~/datasets/martin_room
CONFIG_PATH=configs/martin_room.json

echo training on $CKPT_PATH

python opt.py -t $CKPT_PATH $DATASET_PATH -c $CONFIG_PATH

# python render_imgs.py $CKPT_PATH/ckpt.npz $DATASET_PATH -c $CONFIG_PATH

cp $CONFIG_PATH $CKPT_PATH/config.json