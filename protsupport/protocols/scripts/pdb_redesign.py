import sys
import os
import time
from pyrosetta import *

from torchsupport.data.io import netread

from protsupport.protocols.minimization_packing.design import (
  NetPackMover, AnnealedNetPackMover
)
from protsupport.modules.baseline import Baseline

if __name__ == "__main__":
  init("-mute all")
  pose = pose_from_file(sys.argv[1])
  net = Baseline(aa_size=20, in_size=39, hidden_size=100, neighbours=15) 
  netread(net, sys.argv[3])
  net.eval()
  fixed_path = sys.argv[2]
  with open(fixed_path) as fx:
    fix_list = []
    for line in fx:
      fix_list.append(int(line.strip()) - 26)
    fix_list = sorted(list(set(fix_list)))
  scorefxn = get_fa_scorefxn()
  mover = NetPackMover(
    net, pose, fix=fix_list, glycinate=True,
    scorefxn=scorefxn, kT=0.1, max_iter=1000
  )
  # mover = AnnealedNetPackMover(
  #   net, pose, fix=fix_list, glycinate=True, n_moves=50,
  #   scorefxn=scorefxn, kT_high=100.0, kT_low=0.3, max_iter=100
  # )

  out_path = sys.argv[4]
  index = 0
  
  while os.path.isfile(os.path.join(out_path, f"log-{index}")):
    time.sleep(1)
    index += 1
  with open(os.path.join(out_path, f"log-{index}"), "w") as log:
    step = 0
    while True:
      mover.apply(pose)
      pose.dump_pdb(os.path.join(out_path, f"candidate-{index}-step-{step}.pdb"))
      value = scorefxn.score(pose)
      sequence = mover.monte_carlo.lowest_score_pose().sequence()
      log.write(f"{step}\t{sequence}\t{value}\n")
      step += 1
      log.flush()
