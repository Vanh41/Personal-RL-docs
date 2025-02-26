m = int(input())
n = int(input())
tmp = []
newlist = []
for i in range(m):
    ele = input()
    tmp.append(ele)

for i in range(1,n+1):
    for ele in tmp:
        ele = ele + str(i)
        newlist.append(ele)

print(newlist)