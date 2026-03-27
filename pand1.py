import pandas as pd
# print("read data form the head")
# print(df.head(10))
# print("display the data from the tail"
#       )
# print(df.tail(10))
data={
    "Name":["Alice","Bob","Charlie","David","Eve"],
    "Age":[25,30,35,40,45],
    "salary":[50000,60000,70000,80000,90000],
    "performance score":[85,90,95,80,75]

}
df=pd.DataFrame(data)
# print(df)
print(df["Name"])
print(df[["Name","Age"]] )
print(df[df["Age"]>30])