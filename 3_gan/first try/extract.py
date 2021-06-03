import pickle
from cv2 import cv2
with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

x_train = []
y_train = []
x_test = []
y_test = []

limit = 900
count = 0
for i in dataset:
    vec = i[0]
    img = i[1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if count<limit:
        x_train.append(img)
        y_train.append(vec)
        count += 1
    else:
        x_test.append(img)
        y_test.append(vec)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

data = []
data.append(x_train)
data.append(y_train)
data.append(x_test)
data.append(y_test)

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
