import os

totalBandwidth = 0
totalTX = 0
for root,dirs,files in os.walk("./regtest"):
    for f in files:
        if f=="debug.log":
            fullname = os.path.join(root, f)
            #print(fullname)
            with open(fullname) as data:
                flag =0
                cnt = 0
                for line in reversed(list(data)):
                     info = line.rstrip()
                     if flag==0 and "bandwidth" in info:
                         #print(info[37:])
                         totalBandwidth+=int(info[37:])
                         flag = 2
                     if "AcceptTo" in info:
                         cnt+=1
            print(cnt)
            totalTX+=cnt;
print("bandwidth: "+str(totalBandwidth))
print("transactions: "+str(totalTX))