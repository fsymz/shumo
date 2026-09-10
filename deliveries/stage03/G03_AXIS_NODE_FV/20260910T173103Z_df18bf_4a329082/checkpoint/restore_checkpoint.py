#!/usr/bin/env python3
"""Restore the source/config checkpoint without running it. Keep original packet."""
from pathlib import Path
import argparse,base64,hashlib,json,zlib
EXPECTED='772205df86b8fc4458d16e47a513f2d2fa37a99cb39351a019acb76b9ead4c11'
DECODED='7361f2f5db1b82bb17d77f7cb8215f9caadaf4bfc1393f1d2ac288c2544542f3'
def main():
 p=argparse.ArgumentParser();p.add_argument('file',type=Path,nargs='?',default=Path('solver_checkpoint.json.zlib.b64'));p.add_argument('--output',type=Path,default=Path('recovered_solver'));a=p.parse_args()
 b=a.file.read_bytes()
 if len(b)!=18021 or hashlib.sha256(b).hexdigest()!=EXPECTED:raise ValueError('CHECKPOINT_SIZE_OR_SHA256_MISMATCH')
 raw=zlib.decompress(base64.b64decode(b,validate=False))
 if hashlib.sha256(raw).hexdigest()!=DECODED:raise ValueError('DECODED_SHA256_MISMATCH')
 x=json.loads(raw);root=a.output.resolve();pending=[]
 for f in x['files']:
  target=(root/f['path']).resolve();data=f['text'].encode()
  if not target.is_relative_to(root):raise ValueError('UNSAFE_PATH')
  if hashlib.sha256(data).hexdigest()!=f['sha256']:raise ValueError('SOURCE_SHA256_MISMATCH')
  if target.exists() and target.read_bytes()!=data:raise FileExistsError(target)
  pending.append((target,data))
 for target,data in pending:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 print(json.dumps({'request_id':x['request_id'],'group_id':x['group_id'],'files_restored':len(pending),'all_sha256_verified':True,'code_executed':False,'full_delivery':False,'required_original_inputs':'Retain original task packet; this is only the solver/config checkpoint.'}))
if __name__=='__main__':main()
