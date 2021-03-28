import os

totalBandwidth = 0
for root,dirs,files in os.walk("./regtest_100"):
    for f in files:
        if f=="debug.log":
            fullname = os.path.join(root, f)
            #print(fullname)
            with open(fullname) as data:
                for line in reversed(list(data)):
                     info = line.rstrip()
                     if "bandwidth" in info:
                         #print(info[37:])
                         totalBandwidth+=int(info[37:])
                         break;
            break;
print(totalBandwidth)