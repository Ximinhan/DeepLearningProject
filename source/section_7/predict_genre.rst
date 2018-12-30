用深度学习利用电影海报预测电影类型
========================================

我们必须再次做出实现决策。这一次，更多的是关于我们愿意花费多少时间来换取更高的准确性。我们将在这里使用一种在机器学习教程中通常被称为预训练的技术。

为了防止我在这里重新造轮子，我打算借用斯坦福大学卷积神经网络 课程_ 中的一段介绍。

.. _课程: http://cs231n.github.io/transfer-learning/

引用——“在实践中，很少有人从头开始训练整个卷积神经网络(使用随机初始化),因为很少情况下能够拥有足够大的数据集。相反，在一个非常大的数据集(例如，ImageNet上包含了120万张具有1000个类别的图像)预训练卷积网络是很常见的，然后这个卷积网络作为目标任务的初始化或者固定特征提取工具。”

有三种广泛的方法可以完成迁移学习或者预训练。(这两个概念是不同的，为了清楚的理解这两个概念之间的区别，我建议你仔细阅读相关的教程)。我们将要做的是使用一个预训练过的、发布过的卷积网络作为特征提取工具。从ImageNet(一个非常流行的对象检测数据集)上获取一个预训练过的卷积网络模型，然后移除最后一个全连接层。去掉最后一层后，我们得到的是另一个神经网络即一堆空间转换。但是，原来这个栈的输出可以输入到一个单独的层中，这个层可以对图片分类比如汽车、狗、猫等等。

这就意味着，在这个空间层中，神经网络将所有图片中包含“狗”的图片聚集在一起，所有包含“猫”的图片聚集在一起。因此，这是一个很有用的层，它使得具有相似对象的图片逻辑距离更相近。

设想一下，如果我们将我们的电影海报输入到这个栈中，它会将它们嵌入到这样一个空间中，其中具有相似对象的海报逻辑距离更相近。这是一种非常有意义的特征工程方法！尽管也许这并不是类型预测的理想方法，但这确实很有意义。比方说，那些电影海报中有枪或者汽车的很有可能是动作电影。而那些包含微笑着的夫妇的很有可能是浪漫题材或者戏剧题材。另一种选择是从头开始训练卷积神经网络，这是相当密集的计算并且涉及许多技巧使神经网络的训练收敛到最佳的空间转换。

这样，我们就可以从一些强大的地方开始，然后再建立上层。我们将我们的图片输入到预训练过的网络中来从这些海报中提取视觉特征。然后，用这些特征作为图像的描述，类型作为标签，我们从头创建一个可以在此数据集上学习简单分类的简单的神经网络。这两个步骤正是我们为了预测海报类型要做的。

利用深度学习从海报中提取视觉特征
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

我们要解决的基本问题是我们能否从电影海报预测电影类型。先要确定-这个假设是否有意义。答案是有意义的。因为这正是海报设计者的工作。他们给海报标语留下视觉线索。他们确保当我们看到一个恐怖电影的海报时我们可以知道这不是一个欢乐的画面。像这样的事情我们能否用深度学习系统推断出这样的微妙之处么？让我们看看！

对于视觉特征，要么我们可以从头开始训练一个深层神经网络，要么我们可以使用牛津大学视觉几何学组提供给我们的一个预先训练的神经网络，这是最流行的方法之一。这就是所谓的VGG网络。或者正如他们所说，我们将提取图像的VGG特性。在数学上，如前所述，它只是一个层形式的空间变换。所以，我们只需要在图像上执行这一系列的转换，对吧？Keras是一个库可以很容易让我们做到这一点。还有一些其他的库如Tensorflow和PyTorch。尽管后两者功能很强大也可定制并且在实践中用的更多，但Keras通过保持简单的语法使得更容易实现原型。

我们将使用Keras使得代码简单化，这样我们可以花费更多时间理解和更少的时间编码。对于这一步人们有许多常见做法——“获取图像的VGG特征”或是“通过VGG反向传播图片并移除最后一层”。在Keras中，简单的使用4行代码就够了。

::

  # Loading the list of movies we had downloaded posters for eariler - 
  f=open('poster_movies.pckl','r')
  poster_movies=pickle.load(f)
  f.close()

::
  
  from keras.applications.vgg16 import VGG16
  from keras.preprocessing import image
  from keras.applications.vgg16 import preprocess_input
  import numpy as np
  import pickle
  model = VGG16(weights='imagenet', include_top=False)

::
  
  allnames=os.listdir(poster_folder)
  imnames=[j for j in allnames if j.endswith('.jpg')]
  feature_list=[]
  genre_list=[]
  file_order=[]
  print "Starting extracting VGG features for scraped images. This will take time, Please be patient..."
  print "Total images = ",len(imnames)
  failed_files=[]
  succesful_files=[]
  i=0
  for mov in poster_movies:
      i+=1
      mov_name=mov['original_title']
      mov_name1=mov_name.replace(':','/')
      poster_name=mov_name.replace(' ','_')+'.jpg'
      if poster_name in imnames:
          img_path=poster_folder+poster_name
          try:
              img = image.load_img(img_path, target_size=(224, 224))
              succesful_files.append(imname)
              x = image.img_to_array(img)
              x = np.expand_dims(x, axis=0)
              x = preprocess_input(x)
              features = model.predict(x)
              file_order.append(img_path)
              feature_list.append(features)
              genre_list.append(mov['genre_ids'])
              if np.max(np.asarray(feature_list))==0.0:
                  print('problematic',i)
              if i%250==0 or i==1:
                  print "Working on Image : ",i
          except:
              failed_files.append(imname)
              continue
        
      else:
          continue
  print "Done with all features, please pickle for future use!"

Starting extracting VGG features for scraped images. This will take time, Please be patient...

Total images =  1347

Working on Image :  250

Working on Image :  500

Working on Image :  750

Working on Image :  1000

Working on Image :  1250

Working on Image :  1500

Done with all features, please pickle for future use!

::
  
  len(genre_list)

1317

::
  
  len(feature_list)

1317

::
  
  # Reading from pickle below, this code is not to be run.
  list_pickled=(feature_list,file_order,failed_files,succesful_files,genre_list)
  f=open('posters_new_features.pckl','wb')
  pickle.dump(list_pickled,f)
  f.close()
  print("Features dumped to pickle file")

Features dumped to pickle file

::
  
  f7=open('posters_new_features.pckl','rb')
  list_pickled=pickle.load(f7)
  f7.close()
  # (feature_list2,file_order2)=list_pickled

使用VGG特征训练一个简单神经网络模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::
  
  (feature_list,files,failed,succesful,genre_list)=list_pickled

让我们首先获取我们1342个样本的标签！由于一些实例下载图片失败，使用正确模型的最佳方法是读取下载的海报的名称，然后从那里开始。这些海报无法上传到Github上因为他们太大了，因此他们将被下载下来然后从我电脑本地读取。如果你确定要重做，你可能要检查和修改代码中的路径以确保可以正确运行。

::
  
  (a,b,c,d)=feature_list[0].shape
  feature_size=a*b*c*d

这看起来很奇怪，为什么要再次执行上面执行过的循环？原因很简单，关于numpy最重要的是使用vstack()和hstack()是高度次优的。当Numpy数组创建后会在内从中分配一个固定的大小，当我们进行stack操作时会在一个新的位置复制并创建一个新的数组。这使得代码执行非常非常慢。解决这个问题最好的方法(如果你使用的是MATLAB矩阵也存在这个问题)是创建一个零组成的numpy数组然后逐行重写它。上面的代码只是用来看看我们需要多大的numpy数组！

最终的电影海报数据集，我们所需要的所有信息，是1265部电影。在上面的代码中，我们制作了一个X numpy数组，其中包含每行一个图像的视觉特性。因此，将vgg特征整形为形状(1,25088)，最后得到形状矩阵(1265,25088)。

::
 
  np_features=np.zeros((len(feature_list),feature_size))
  for i in range(len(feature_list)):
      feat=feature_list[i]
      reshaped_feat=feat.reshape(1,-1)
      np_features[i]=reshaped_feat

  X=np_features

  from sklearn.preprocessing import MultiLabelBinarizer
  mlb=MultiLabelBinarizer()
  Y=mlb.fit_transform(genre_list)

  Y.shape

(1317, 20)

我们布尔型的的Y numpy数组包含与1277部电影的流派ID相对应的布尔型标签。

::
  
  visual_problem_data=(X,Y)
  f8=open('visual_problem_data_clean.pckl','wb')
  pickle.dump(visual_problem_data,f8)
  f8.close()

  f8=open('visual_problem_data_clean.pckl','rb')
  visual_features=pickle.load(f8)
  f8.close()

  (X,Y)=visual_features

  X.shape

  mask = np.random.rand(len(X)) < 0.8

  X_train=X[mask]
  X_test=X[~mask]
  Y_train=Y[mask]
  Y_test=Y[~mask]

  X_test.shape

(264, 25088)

现在，我们创建了自己的keras神经网络来使用vgg特性，然后对电影类型进行分类。Keras让这变得非常容易。

多年来，神经网络结构变得越来越复杂。但是最简单的计算包含了非常标准的分层计算，如上所述。考虑到其中一些操作的流行性，Keras可以很容易地按顺序写出这些操作的名称。这样你就可以在完全避免数学的同时建立一个网络（强烈建议在数学上花更多的时间）

sequential（）允许我们使模型遵循这个层的顺序。可以使用不同类型的层，如致密层、conv2d层等，还可以使用许多激活功能，如relu、linear等。

