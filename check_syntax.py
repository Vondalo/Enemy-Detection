import ast

# Read the file
with open('c:/Users/richi/Enemy-Dectection/src/process_video_improved.py', 'r') as f:
    source = f.read()

# Try to parse it
try:
    ast.parse(source)
    print("Syntax is valid")
except SyntaxError as e:
    print(f"Syntax error: {e}")
    print(f"Line {e.lineno}: {e.text}")
