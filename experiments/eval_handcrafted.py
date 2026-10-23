import glob, torch
from backdoor import dataset
from backdoor.badnet import BadNetDataPoisoning, Trigger
from backdoor.training import Trainer

ds = dataset.GTSB()
data = ds.get_data()
trigger = Trigger.from_string('checkerboard("bottomleft",(3,3),colours=(255,0))')
badnet = BadNetDataPoisoning.always_backdoor(trigger, backdoor_class=6)
test_bd = badnet.apply(data['test'], poison_only=True)

rows = []
for w in sorted(glob.glob('/home/wyl/backdoor/experiments/weights/gtsb_handcrafted/*.pth')):
    model = torch.load(w, map_location='cuda')
    t = Trainer(model, use_wandb=False, device='cuda')
    ts = t.evaluate_epoch(*data['test'], bs=512, name='test_eval', progress_bar=False)
    bd = t.evaluate_epoch(*test_bd, bs=512, name='test_bd', progress_bar=False)
    rows.append((w.split('/')[-1], ts['test_eval_acc'], bd['test_bd_acc']))

rows.sort(key=lambda r: -r[2])
print(f"{'权重文件':<42}{'test_acc':>10}{'test_bd_acc':>12}")
for name, acc, bd in rows:
    print(f"{name:<42}{acc:>10.4f}{bd:>12.4f}")
