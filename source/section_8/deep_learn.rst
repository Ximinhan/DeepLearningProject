利用深度学习获得文本特征
=========================

让我们试着对文本做同样的事？

我们将使用现成的文字表达式模型-Word2Vec模型。就像之前的VGGnet一样，这个模型可以获得有意义的表达式。由于单词总数很小，我们甚至不需要通过网络来传递我们的样本。即使这样我们也做了，并且将结果以字典的形式保存。我们可以简单的通过查询字典获取单词的Word2Vec特征。

你可以在这下载字典-https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit 

下载字典到当前同级目录下。

::
  
  from gensim import models
  # model2 = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
  model2 = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

现在我们只需要简单的从上面的模型中查找单词就行了。比方说，我们要获取单词“King”的Word2Vec特征值只需要查找-model2['king']

::
  
  print model2['king'].shape
  print model2['dog'].shape

  (300,)
  (300,)

这样一来，我们就能用这个word2vec模型代表我们概述中的单词了。然后我们可以以此作为我们的X参数。因此，我们使用的是基于单词语义表示的表示，而不是字数。从数学上讲，每个单词可以从3-4维（长度）到300维！

对于上面的电影数据集，我们试试通过概述内容预测一下类型！

::
  
  len(final_movies_set)

1265

::
  
  from nltk.tokenize import RegexpTokenizer
  from stop_words import get_stop_words
  tokenizer = RegexpTokenizer(r'\w+')

  # create English stop words list
  en_stop = get_stop_words('en')

::
  
  movie_mean_wordvec=np.zeros((len(final_movies_set),300))
  movie_mean_wordvec.shape

(1265,300)

这样，我们就可以使用这个word2vec模型来表示概述中的单词。然后，我们可以用它作为x表示。因此，我们使用的是基于单词语义表示的表示，而不是字数。从数学上讲，每个单词从3-4维（长度）到300维！

对于上面的同一组电影，让我们试着从它们的概述的深层表现来预测类型！

::
  
  len(final_movies_set)

1265

::
  
  from nltk.tokenize import RegexpTokenizer
  from stop_words import get_stop_words
  tokenizer = RegexpTokenizer(r'\w+')

  # create English stop words list
  en_stop = get_stop_words('en')

::
  
  movie_mean_wordvec=np.zeros((len(final_movies_set),300))
  movie_mean_wordvec.shape

(1265,300)

在我们对文本训练模型之前需要做一些预处理。我们要做的唯一的事就是——删除我们知道的没有特殊含义的常见单词。某种意义上来说，这对我们来说就是噪音。这些单词经常被删除并且被称为“stop words”。你可以在网上查到他们。其中包括“a”、“and”、“but”、“how”、“or”等简单单词。使用python包nltk可以很容易地删除它们。

在上述数据集中，仅包含“stop words”的电影概述的电影和仅有word2vec表达式的概述的电影会被忽略。其余的被用于构建我们的Mean word2vec表达式。简单来说，对每部电影概述——

 - 获取电影概述
 - 删掉其中的stop words
 - 如果属于word2vec 那么取它的300维的word2vec标识
 - 如果不是 输出单词
 - 对每部电影来说，对所有概述中没有被输出的单词计算300维向量表达式的平均值

这意味着它将成为电影的300维表示。对于所有电影，这些都存储在一个numpy数组中。因此X矩阵变成(1263,300)，而且Y变成(1263,20)，也就是说之前说过的二值化的20中类型。

为什么取算术平均数？如果你觉得我们应该把所有的单词分开——那么你的想法是正确的，但遗憾的是，我们受到了当今神经网络工作方式的限制。我不会考虑这个问题，因为担心在其他不相关的细节上压力过大。但如果你感兴趣，请阅读这篇精彩的论文-https://jiajunwu.com/papers/dmil_cvpr.pdf

::

  genres=[]
  rows_to_delete=[]
  for i in range(len(final_movies_set)):
      mov=final_movies_set[i]
      movie_genres=mov['genre_ids']
      genres.append(movie_genres)
      overview=mov['overview']
      tokens = tokenizer.tokenize(overview)
      stopped_tokens = [k for k in tokens if not k in en_stop]
      count_in_vocab=0
      s=0
      if len(stopped_tokens)==0:
          rows_to_delete.append(i)
          genres.pop(-1)
  #         print overview
  #         print "sample ",i,"had no nonstops"
      else:
          for tok in stopped_tokens:
              if tok.lower() in model2.vocab:
                  count_in_vocab+=1
                  s+=model2[tok.lower()]
          if count_in_vocab!=0:
              movie_mean_wordvec[i]=s/float(count_in_vocab)
          else:
              rows_to_delete.append(i)
              genres.pop(-1)
  #             print overview
  #             print "sample ",i,"had no word2vec"

::

  len(genres)

  1261

::
  
  mask2=[]
  for row in range(len(movie_mean_wordvec)):
      if row in rows_to_delete:
          mask2.append(False)
      else:
          mask2.append(True)

::
  
  X=movie_mean_wordvec[mask2]

::
  
  X.shape

(1261,300)

::
  
  Y=mlb.fit_transform(genres)

::
  
  Y.shape

(1261,20)

::
  
  textual_features=(X,Y)
  f9=open('textual_features.pckl','wb')
  pickle.dump(textual_features,f9)
  f9.close()

::
  
  # textual_features=(X,Y)
  f9=open('textual_features.pckl','rb')
  textual_features=pickle.load(f9)
  f9.close()

::
  
  (X,Y)=textual_features

::
  
  X.shape

(1261,300)

::
  
  Y.shape

(1261,20)

::
  
  mask_text=np.random.rand(len(X))<0.8
  
  X_train=X[mask_text]
  Y_train=Y[mask_text]
  X_test=X[~mask_text]
  Y_test=Y[~mask_text]

我们再一次使用与之前相似的那种简单架构

::

  from keras.models import Sequential
  from keras.layers import Dense, Activation

  model_textual = Sequential([
      Dense(300, input_shape=(300,)),
      Activation('relu'),
      Dense(20),
      Activation('softmax'),
  ])

  model_textual.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

  model_textual.fit(X_train, Y_train, epochs=10, batch_size=500)

Epoch 1/10

982/982 [==============================] - 0s - loss: 0.4819 - acc: 0.8520     

Epoch 2/10

982/982 [==============================] - 0s - loss: 0.4642 - acc: 0.8520     

Epoch 3/10

982/982 [==============================] - 0s - loss: 0.4527 - acc: 0.8520     

Epoch 4/10

982/982 [==============================] - 0s - loss: 0.4456 - acc: 0.8520     

Epoch 5/10

982/982 [==============================] - 0s - loss: 0.4407 - acc: 0.8520     

Epoch 6/10

982/982 [==============================] - 0s - loss: 0.4367 - acc: 0.8520     

Epoch 7/10

982/982 [==============================] - 0s - loss: 0.4332 - acc: 0.8520     

Epoch 8/10

982/982 [==============================] - 0s - loss: 0.4295 - acc: 0.8520     

Epoch 9/10

982/982 [==============================] - 0s - loss: 0.4260 - acc: 0.8520     

Epoch 10/10

982/982 [==============================] - 0s - loss: 0.4227 - acc: 0.8520   

<keras.callbacks.History at 0x4e27e3850>

::

  model_textual.fit(X_train, Y_train, epochs=10000, batch_size=500,verbose=0)

<keras.callbacks.History at 0x4e27e3a10>

::
  
  score = model_textual.evaluate(X_test, Y_test, batch_size=249)

249/279 [=========================>....] - ETA: 0s

::
  
  print("%s: %.2f%%" % (model_textual.metrics_names[1], score[1]*100))

acc: 86.52%

::
  
  Y_preds=model_textual.predict(X_test)
  genre_list.append(10769)

  print "Our predictions for the movies are - \n"
  precs=[]
  recs=[]
  for i in range(len(Y_preds)):
      row=Y_preds[i]
      gt_genres=Y_test[i]
      gt_genre_names=[]
      for j in range(20):
          if gt_genres[j]==1:
              gt_genre_names.append(Genre_ID_to_name[genre_list[j]])
      top_3=np.argsort(row)[-3:]
      predicted_genres=[]
      for genre in top_3:
          predicted_genres.append(Genre_ID_to_name[genre_list[genre]])
      (precision,recall)=precision_recall(gt_genre_names,predicted_genres)
      precs.append(precision)
      recs.append(recall)
      if i%50==0:
          print "Predicted: ",predicted_genres," Actual: ",gt_genre_names

Our predictions for the movies are - 

Predicted:  [u'Science Fiction', u'Action', u'Adventure']  Actual:  [u'Adventure', u'Action', u'Comedy', u'Romance']

Predicted:  [u'Thriller', u'Crime', u'Mystery']  Actual:  [u'Drama', u'Thriller', u'Science Fiction']

Predicted:  [u'Action', u'Crime', u'Thriller']  Actual:  [u'Adventure', u'Action', u'Comedy', u'Thriller', u'Crime']

Predicted:  [u'Family', u'Horror', u'Comedy']  Actual:  [u'Horror', u'Action', u'Thriller']

Predicted:  [u'Crime', u'Thriller', u'Drama']  Actual:  [u'Action', u'Science Fiction']

Predicted:  [u'Drama', u'Thriller', u'Mystery']  Actual:  [u'Drama', u'Thriller', u'Mystery', u'Romance']

::
  
  print np.mean(np.asarray(precs)),np.mean(np.asarray(recs))

0.519713261649 0.563918757467

即使对上面的模型没有太多调整，这些结果也能够超越我们之前的结果。

注-当我对从维基百科中的爬取的情节进行分类时，我的准确率高达78%。大量的信息非常适合用深度模型对电影类型进行分类。强烈建议您尝试使用这种架构。
