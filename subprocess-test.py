import subprocess

res = subprocess.run('C:\\tool\\paligemma\\dist\\test2\\test2.exe', shell=True, capture_output=True, text=True)
print("kekka)", res)

