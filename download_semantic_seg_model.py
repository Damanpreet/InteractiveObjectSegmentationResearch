import yaml
import os
from six.moves import urllib

download_cfg = yaml.safe_load(open('semantic_config.yaml'))

url = download_cfg["DATASET"]["DOWNLOAD_URL"]
model_path = download_cfg['DATASET']['DOWNLOAD_DIR']
tar_name = download_cfg["DATASET"]["TARBALL_NAME"]

if not os.path.exists(model_path):
    os.makedirs(model_path)

download_path = os.path.join(model_path, tar_name) 
print(download_path)
urllib.request.urlretrieve(url, download_path)	
print("download completed!")

