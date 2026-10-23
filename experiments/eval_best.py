import torch
from backdoor import dataset
from backdoor.badnet import BadNetDataPoisoning, Trigger
from backdoor.training import Trainer

ds = dataset.GTSB()
data = ds.get_data()
trigger = Trigger.from_string('checkerboard("bottomleft",(3,3),colours=(255,0))')
badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=6)
test_bd = badnet.apply(data['test'], poison_only=True)

for w in ['/home/wyl/backdoor/experiments/weights/gtsb_handcrafted/tm2_v3:handcrafted_fe1522.pth',
          '/home/wyl/backdoor/experiments/weights/gtsb_handcrafted/tm2_v3:handcrafted_d8be54.pth']:
    model = torch.load(w, map_location='cuda')
    t = Trainer(model, use_wandb=False, device='cuda')
    ts = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
    bd = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
    neg = t.evaluate_epoch(test_bd.X, data['test'].y, bs=512, name='test_bd_neg', progress_bar=False)
    print(w.split('/')[-1], '| acc:', round(ts['test_eval_acc'],4), '| bd:', round(bd['test_bd_acc'],4), '| neg:', round(neg['test_bd_neg_acc'],4))
