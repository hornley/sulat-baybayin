import os
root='single_symbol_data'
exts=('.png','.jpg','.jpeg','.bmp')
if not os.path.isdir(root):
    print('No data folder found')
    raise SystemExit(1)
counts={}
total=0
for entry in sorted(os.listdir(root)):
    p=os.path.join(root,entry)
    if os.path.isdir(p):
        imgs=[f for f in os.listdir(p) if f.lower().endswith(exts)]
        counts[entry]=len(imgs)
        total+=len(imgs)
print(f'Found {len(counts)} classes and {total} images')
for k in sorted(counts):
    v=counts[k]
    note=''
    if v<2:
        note='  <-- WARNING: <2 images (stratified split will fail)'
    elif v<10:
        note='  <-- few images (consider augmenting)'
    print(f'  {k}: {v}{note}')
