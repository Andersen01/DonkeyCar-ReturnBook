import os

import shutil

path = './tub'

new_path = './test_images'


'''
for root, dirs, files in os.walk(path):#计算图片总数
    for i in range(len(files)):
     if (files[i][-3:] == 'jpg'):
      count=count+1

min=count-49
print(count)
print(min)
num=0
for root, dirs, files in os.walk(path):
  for i in range(len(files)):
     if (files[i][-3:] == 'jpg'):
       num = num + 1
       if(num>=min):
            print(files[i])
            a=files[i].split('_', 1)[0]
            a=int(a)
            print(a)
            file_path = root + '/' + files[i]
            new_file_path = new_path + '/' + files[i]
            shutil.copy(file_path, new_file_path)
            num=num+1'''

MAX=0
for root, dirs, files in os.walk(path):#返回图片最大值
    for i in range(len(files)):
     if (files[i][-3:] == 'jpg'):
         a = files[i].split('_', 1)[0]
         a = int(a)
         if(MAX<a):
             MAX=a

MIN=MAX-49

for root, dirs, files in os.walk(path):#保存图片
  for i in range(len(files)):
     if (files[i][-3:] == 'jpg'):
            num=files[i].split('_', 1)[0]
            num=int(num)
            if (num<=MAX and MIN<=num):
              file_path = root + '/' + files[i]
              new_file_path = new_path + '/' + files[i]
              shutil.copy(file_path, new_file_path)


