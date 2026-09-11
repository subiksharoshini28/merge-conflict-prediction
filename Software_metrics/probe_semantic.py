import subprocess, tempfile, os, tarfile, io
REPO = "D:/commons-lang"
def g(*a): return subprocess.run(["git","-C",REPO,*a],capture_output=True,text=True)

def compiles(tree):
    d = tempfile.mkdtemp()
    ar = subprocess.run(["git","-C",REPO,"archive","--format=tar",tree,"src/main/java"],capture_output=True)
    if ar.returncode != 0 or not ar.stdout: return None,"no src/main/java"
    tarfile.open(fileobj=io.BytesIO(ar.stdout)).extractall(d)
    srcs = [os.path.join(r,f) for r,_,fs in os.walk(d) for f in fs if f.endswith(".java")]
    if not srcs: return None,"no java files"
    lst = os.path.join(d,"srcs.txt")
    open(lst,"w").write("\n".join(srcs))
    r = subprocess.run(["javac","-d",os.path.join(d,"out"),"-proc:none","@"+lst],capture_output=True,text=True)
    return (r.returncode==0), (r.stderr[:160] if r.returncode else "OK")

merges = g("log","--merges","--min-parents=2","--max-parents=2","--format=%H").stdout.split()
print("total 2-parent merges:", len(merges))
tested = 0
for m in merges:
    ps = g("rev-list","--parents","-n","1",m).stdout.split()[1:]
    if len(ps) != 2: continue
    p1,p2 = ps
    mt = subprocess.run(["git","-C",REPO,"merge-tree","--write-tree",p1,p2],capture_output=True,text=True)
    if mt.returncode != 0:
        print(m[:8], "-> textual conflict, skip"); continue
    tree = mt.stdout.strip().splitlines()[0]
    okm, msg = compiles(tree)
    print(f"{m[:8]} clean-merge merged-compiles={okm}  {msg.strip()[:70]}")
    tested += 1
    if tested >= 5: break
