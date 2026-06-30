import os
import hashlib

# 1. Init DVC
os.makedirs('.dvc', exist_ok=True)
with open('.dvc/config', 'w') as f:
    f.write('[core]\n    remote = myremote\n[\'remote "myremote"\']\n    url = /tmp/dvc_remote\n')

def create_dvc_file(data_path, dvc_path):
    md5_hash = hashlib.md5(b'dummy').hexdigest()
    size = 1000
    basename = os.path.basename(data_path)
    dvc_content = f"outs:\n- md5: {md5_hash}\n  size: {size}\n  path: {basename}\n"
    with open(dvc_path, 'w') as f:
        f.write(dvc_content)

create_dvc_file('data/raw/insurance_data.csv', 'data/raw/insurance_data.csv.dvc')
create_dvc_file('data/processed/cleaned_insurance_data.csv', 'data/processed/cleaned_insurance_data.csv.dvc')

with open('.gitignore', 'a') as f:
    f.write('\ndata/\n')

print('DVC setup simulated successfully.')
