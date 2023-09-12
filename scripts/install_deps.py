import shutil

import scripts.yaml as yaml

def main():
    shutil.copy("../docker-compose.yaml.template",
        "../docker-compose.yaml")

if __name__ == "__main__":
    main()
