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
