import json
import os

all_jobs = []

for root, dirs, files in os.walk("./jobs/", topdown=False):
   for name in files:
      print(os.path.join(root, name))
      with open(os.path.join(root, name), "r") as f:
          all_jobs.append(json.loads(f.read()))

with open("data/jobs.json", "w") as f:
    f.write(json.dumps(all_jobs))