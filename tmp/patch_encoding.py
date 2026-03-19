import os

os.makedirs(r"C:\Users\richi\Enemy-Dectection\tmp", exist_ok=True)

files_to_patch = [
    r"C:\Users\richi\Enemy-Dectection\src\train.py",
    r"C:\Users\richi\Enemy-Dectection\src\test_gpu.py",
    r"C:\Users\richi\Enemy-Dectection\src\screenshot_tool.py",
    r"C:\Users\richi\Enemy-Dectection\src\model.py",
    r"C:\Users\richi\Enemy-Dectection\SETUP.py",
    r"C:\Users\richi\Enemy-Dectection\analyze_positional_bias.py",
    r"C:\Users\richi\Enemy-Dectection\src\augment_dataset_improved.py",
    r"C:\Users\richi\Enemy-Dectection\src\process_video_improved.py",
    r"C:\Users\richi\Enemy-Dectection\src\visualize_dataset.py",
    r"C:\Users\richi\Enemy-Dectection\src\split_dataset.py",
]

patch = """
# Ensure utf-8 output to avoid charmap encode errors on Windows
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
"""

for fpath in files_to_patch:
    if not os.path.exists(fpath): continue
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "sys.stdout.reconfigure" in content:
        continue
        
    lines = content.split('\n')
    
    # insert after the last import
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_idx = i + 1
            
    if insert_idx == 0:
        # fallback to line 1 if no imports found, but skip shebang if present
        if lines and lines[0].startswith('#!'):
            insert_idx = 1
            
    lines.insert(insert_idx, patch)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Patched {os.path.basename(fpath)}")
