net_name = "camvid720x720_ATONet_final3_loss3_5_"
norm = 'BN_'
bins = (8, 4, 2)
use_ohem = False
batch = 16
epoch = 400
info = net_name + norm + "batch={}_".format(str(batch)) \
           + "use_ohem={}_".format(str(use_ohem)) + "bins={}".format('_'.join(list(map(str, bins)))) \
           + "epoch={}".format(str(epoch))