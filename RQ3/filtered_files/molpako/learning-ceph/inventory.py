from pathlib import Path
import json
import subprocess
import sys
import os
from collections import defaultdict

# multipassのVMのリストを取得するコマンド
cmd = ['multipass', 'list', '--format', 'json']
result = subprocess.run(cmd, stdout=subprocess.PIPE)

# multipassコマンドの出力をJSONデータに変換する
instances = json.loads(result.stdout.decode('utf-8'))
inventory = {
    "all": {
        "vars": {
            'ansible_port': 22,
            "ansible_ssh_user": "ubuntu",
            'ansible_ssh_private_key_file': str(Path(os.path.dirname(__file__))/"multipass.key"),
        }
    },
    "clients": {
        "children": ["mons", "mgrs", "osds"],
    },
    "mons": {
        "hosts": [],
    },
    "osds": {
        "hosts": [],
    },
    "admin": {
        "hosts": [],
    },
    "_meta": {
        "hostvars": defaultdict(dict),
    },
}

_meta_vars = inventory['_meta']['hostvars']
for instance in instances["list"]:
    name = instance["name"]
    key = f"{name[:3]}s"
    if name == "mon1":
        key = "admin"
    inventory[key]["hosts"].append(name)
    _meta_vars[name]["ansible_host"] = instance['ipv4'][0]

# Write by ChatGPT-3
if len(sys.argv) == 2 and sys.argv[1] == '--list' or len(sys.argv) < 2:
    # --listフラグが渡された場合、ホストリスト全体を出力する
    print(json.dumps(inventory))
elif len(sys.argv) == 3 and sys.argv[1] == '--host':
    # --hostフラグが渡された場合、指定されたホストの情報を出力する
    hostname = sys.argv[2]
    host = next((hostname for instance in instances["list"] if instance["name"] == hostname), None)
    if host:
        print(json.dumps({'_meta': {'hostvars': _meta_vars[host]}, 'host': host}))
    else:
        print(json.dumps({'_meta': {'hostvars': {}}, 'host': {}}))
else:
    # それ以外の場合は空のJSONを出力する
    print(json.dumps({'_meta': {'hostvars': {}}, 'all': {'hosts': []}}))
