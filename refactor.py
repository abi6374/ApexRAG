import os
import glob

# 1. Rename 'src' to 'apex_rag'
if os.path.exists('src'):
    os.rename('src', 'apex_rag')
    print("Renamed src/ to apex_rag/")

# 2. Find and replace in Python files
files = (
    glob.glob('apex_rag/**/*.py', recursive=True) + 
    glob.glob('tests/**/*.py', recursive=True) + 
    glob.glob('examples/**/*.py', recursive=True)
)

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content.replace('from src.', 'from apex_rag.')
    new_content = new_content.replace('from src ', 'from apex_rag ')
    new_content = new_content.replace('import src.', 'import apex_rag.')
    
    if new_content != content:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {file}')

# 3. Update pyproject.toml
with open('pyproject.toml', 'r', encoding='utf-8') as f:
    content = f.read()

new_content = content.replace('packages = ["src"]', 'packages = ["apex_rag"]')
new_content = new_content.replace('source = ["src"]', 'source = ["apex_rag"]')

if new_content != content:
    with open('pyproject.toml', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Updated pyproject.toml')
