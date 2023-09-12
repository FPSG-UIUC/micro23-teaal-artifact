import os
import sys
sys.path.insert(0, "..")

import shutil

import scripts.yaml as yaml

def main():
    dc_fn = "../docker-compose.yaml"
    shutil.copy(dc_fn + ".template", dc_fn)

    uid = os.getuid()
    gid = os.getgid()

    docker_compose = yaml.load(dc_fn)
    docker_compose["services"]["cl"]["environment"][0] = "USER_UID=" + str(uid)
    docker_compose["services"]["cl"]["environment"][1] = "USER_GID=" + str(gid)
    yaml.dump(docker_compose, dc_fn)

if __name__ == "__main__":
    main()
