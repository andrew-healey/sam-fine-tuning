from glob import glob
import os

ref_limit = 1
for dir in glob("data/**/**/"):
  ref = os.path.join(dir,"ref")
  files = sorted(os.listdir(dir))
  ref_files = files[:ref_limit]

  !mkdir $ref
  for filename in ref_files:
    !mv $dir/$filename $ref/