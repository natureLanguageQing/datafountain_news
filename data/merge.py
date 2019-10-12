import pandas as pd

trainDataSet = pd.read_csv("Train_DataSet.csv").values.tolist()
labelDataSet = pd.read_csv("Train_DataSet_Label.csv").values.tolist()
mergedDataSet = []
for i in trainDataSet:
    message = []
    for j in labelDataSet:
        if i[0] == j[0]:
            string = str(i[1]) + str(i[2])
            message.append(string)
            message.append(j[1])
            if message not in mergedDataSet:
                mergedDataSet.append(message)
# with open("mergedDataSet.csv", "w") as file:
#     file.write("id,title,content,label")
#     for i in mergedDataSet:
#         write_message = ""
#         for z in range(len(i)):
#             if z != len(i) - 1:
#                 write_message += str(i[z])
#                 write_message += ","
#             else:
#                 write_message += str(i[z])
#                 write_message += "\n"
#         file.write(write_message)
test = pd.DataFrame(columns=["text_a", "label"], data=mergedDataSet)  # 数据有三列，列名分别为one,two,three
test.to_csv('merged.csv', encoding='utf-8', index=False)
