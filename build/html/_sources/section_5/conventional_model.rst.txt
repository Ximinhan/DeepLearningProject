利用数据构建非深度的传统机器学习模型
=========================================

下面是我们要做的事-

 - 我们将实现两种不同的模型
 - 我们将做一个性能的度量，即一个定量的方法，以确定两种模型具体的差距如何
 - 讨论一些两种模型的不同之处，比如他们的优点，缺陷等等

前文已经说过，要实现起来需要做许多决策。比如在特性管理、参数调优、模型选择以及您希望模型的可解释性之间（参考:Bayesian vs.-Bayesian方法），有许多问题需要决策。例如，下面是一些可能的模型：

 - 广义线性模型
 - 支持向量机
 - 浅层神经网络
 - 随机森林
 - Boosting算法
 - 决策树

或者是叶贝斯相关的：
 
 - 朴素叶贝斯
 - 线性判别分析
 - 叶贝斯层次模型

这样的列表能一直列下去，但是并不是所有的模型都适用于你自定义框架的问题。你应该想想到底哪个模型最适合你。

对于我们的问题，我将分别从上述两个类别中个选择一个简单的模型作为样例-

 1.支持向量机

 2.多项式朴素叶贝斯

下面是接下来部分的概览：

 - 一些特征管理
 - 两种不同的模型
 - 评估矩阵的选择
 - 模型对比

让我们从一些特征管理开始
~~~~~~~~~~~~~~~~~~~~~~~~~~~

设计正确的特性取决于两个关键思想。首先，你想解决什么问题？例如，如果你想猜我的音乐喜好，并且你试图训练一个超级棒的模型，同时给出我的身高作为输入特征，你将没有运气。另一方面，给它我的Spotify播放列表那么任何模型都能解决问题。因此，问题的内容扮演了一个角色。

其次，您只能根据手头的数据来表示。也就是说，如果你没有访问我的Spotify播放列表，但是访问我的Facebook状态——你知道我所有的关于哈佛的状态可能都没有用。但如果你将我Facebook状态中发的Youtube链接作为输入，那么这也可以解决这个问题。因此，手头数据的可用性是第二个因素。

考虑这个问题的一个好方法是，首先考虑手头的问题，但设计特性受可用数据的限制。如果你有很多独立特征，每个特征都与类相关联，那么学习很容易。另一方面，如果类是一个非常复杂的功能特性，您可能无法学习它。

在这个问题的背景下，我们想预测一部电影的体裁。我们可以查看的是-电影概述，这是电影情节的文本描述。这个假设是有道理的，概括是对故事的简短描述，故事在给电影分配类型方面显然很重要。

因此，让我们通过利用电影概述中的单词来改进我们的特征。回到我们前面讨论的一个有趣的方法-TF-IDF。最初我们用它来过滤单词，但是我们也可以将tf-idf值作为“importance”字段的值赋给单词，而不是认为每个单词权值相等。TF-IDF只是试图给单词组中每个单词赋予一个权重。

同样，它的工作方式是-大多数电影描述都有“The的”这个词。显然，它并没有告诉你任何特别的事情。因此，权重应该与描述单词的电影数量成反比。这是属于IDF的部分。

另一方面，对于电影《星际穿越》来说，如果概述中单词“Space”出现了5次，而单词“wormhole”出现了2次，那么很可能它更多是讲述关于“Space”而不是“wormhole”。因此单词“space”应该有更高的权重。这是属于TF的部分。

我们只是使用TF-IDF给单词组中的每个单词分配权重。这很有道理，是吧？:)

::
  
  from sklearn.feature_extraction.text import TfidfTransformer

  tfidf_transformer = TfidfTransformer()
  X_tfidf = tfidf_transformer.fit_transform(X)
  X_tfidf.shape

(1595, 1365)

让我们把我们的 X 矩阵和 Y 矩阵分成训练部分和测试部分。我们在训练部分中训练模型，然后在训练部分中测出性能。你可以把这个类比成你在做习题集和在考试。当然，他们都是（假设是）来自相同的问题范围。在习题集中取得好成绩是一个很好的指标，表明你在考试中会取得好成绩，但实际上，你必须在你声称你了解这个科目之前进行测试。

::
  
  msk = np.random.rand(X_tfidf.shape[0]) < 0.8

::

  X_train_tfidf=X_tfidf[msk]
  X_test_tfidf=X_tfidf[~msk]
  Y_train=Y[msk]
  Y_test=Y[~msk]
  positions=range(len(movies_with_overviews))
  # print positions
  test_movies=np.asarray(positions)[~msk]
  # test_movies

::
  
  from sklearn.multiclass import OneVsRestClassifier
  from sklearn.svm import SVC
  from sklearn.model_selection import GridSearchCV
  from sklearn.metrics import f1_score
  from sklearn.metrics import make_scorer
  from sklearn.metrics import classification_report

::
  
  parameters = {'kernel':['linear'], 'C':[0.01, 0.1, 1.0]}
  gridCV = GridSearchCV(SVC(class_weight='balanced'), parameters, scoring=make_scorer(f1_score, average='micro'))
  classif = OneVsRestClassifier(gridCV)

  classif.fit(X_train_tfidf, Y_train)

OneVsRestClassifier(estimator=GridSearchCV(cv=None, error_score='raise',estimator=SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False),...refit=True, return_train_score=True,scoring=make_scorer(f1_score, average=micro), verbose=0),n_jobs=1)

::
  
  predstfidf=classif.predict(X_test_tfidf)

  print classification_report(Y_test, predstfidf, target_names=genre_names)

===============       ========= ======    ========   =======
        class         precision recall    f1-score   support
---------------       --------- ------    --------   -------
      Adventure       0.42      0.57      0.48        56
        Fantasy       0.46      0.67      0.55        45
      Animation       0.27      0.42      0.33        31
          Drama       0.60      0.57      0.58       132
         Horror       0.00      0.00      0.00        41
         Action       0.49      0.67      0.57        70
         Comedy       0.40      0.53      0.46        77
        History       0.40      0.37      0.38        27
        Western       0.33      0.14      0.20         7
       Thriller       0.26      1.00      0.41        76
          Crime       0.46      0.48      0.47        46
    Documentary       0.61      0.67      0.64        21
Science Fiction       0.12      1.00      0.22        36
        Mystery       0.23      0.37      0.29        35
          Music       0.95      0.59      0.73        34
        Romance       0.35      0.50      0.41        46
         Family       0.32      0.38      0.35        42
            War       0.27      0.44      0.33         9
        Foreign       0.00      0.00      0.00         4
       TV Movie       0.00      0.00      0.00         5
---------------       --------- ------    --------   -------
    avg / total       0.40      0.56      0.44       840
===============       ========= ======    ========   =======

正如你所看到的，对于像战争和动画这样的表现较少的电影来说，表演总体上更差，而对于像戏剧这样的类别，表现更好。

撇开数字不谈，让我们看看我们的模型对测试集中的一小部分电影的预测。

::

  genre_list=sorted(list(Genre_ID_to_name.keys()))

::
  
  predictions=[]
  for i in range(X_test_tfidf.shape[0]):
      pred_genres=[]
      movie_label_scores=predstfidf[i]
          #print movie_label_scores
      for j in range(20):
          #print j
          if movie_label_scores[j]!=0:
              genre=Genre_ID_to_name[genre_list[j]]
              pred_genres.append(genre)
      predictions.append(pred_genres)

::

  import pickle
  f=open('classifer_svc','wb')
  pickle.dump(classif,f)
  f.close()

::
  
  for i in range(X_test_tfidf.shape[0]):
    if i%50==0 and i!=0:
        print 'MOVIE: ',movies_with_overviews[i]['title'],'\tPREDICTION: ',','.join(predictions[i])

MOVIE:  The Walk 	PREDICTION:  Adventure,Fantasy,Animation,Action,Thriller,Science Fiction

MOVIE:  Cinderella 	PREDICTION:  Adventure,Fantasy,Action,Thriller,Science Fiction

MOVIE:  Liza, the Fox-Fairy 	PREDICTION:  Drama,Thriller,Science Fiction,Romance,War

MOVIE:  The Polar Express 	PREDICTION:  Adventure,Action,Thriller,Science Fiction,Family

MOVIE:  Patema Inverted 	PREDICTION:  Thriller,Science Fiction,Music


让我们看看我们的第二个模型怎么样？朴素叶贝斯模型。

::
  
  from sklearn.naive_bayes import MultinomialNB
  classifnb = OneVsRestClassifier(MultinomialNB())
  classifnb.fit(X[msk].toarray(), Y_train)
  predsnb=classifnb.predict(X[~msk].toarray())

::
  
  import pickle
  f2=open('classifer_nb','wb')
  pickle.dump(classifnb,f2)
  f2.close()

::
  
  predictionsnb=[]
  for i in range(X_test_tfidf.shape[0]):
      pred_genres=[]
      movie_label_scores=predsnb[i]
      for j in range(20):
          #print j
          if movie_label_scores[j]!=0:
              genre=Genre_ID_to_name[genre_list[j]]
              pred_genres.append(genre)
      predictionsnb.append(pred_genres)

::

  for i in range(X_test_tfidf.shape[0]):
    if i%50==0 and i!=0:
        print 'MOVIE: ',movies_with_overviews[i]['title'],'\tPREDICTION: ',','.join(predictionsnb[i])

MOVIE:  The Walk 	PREDICTION:  Adventure,Fantasy,Animation,Science Fiction

MOVIE:  Cinderella 	PREDICTION:  Adventure,Action,Science Fiction

MOVIE:  Liza, the Fox-Fairy 	PREDICTION:  Drama,Romance

MOVIE:  The Polar Express 	PREDICTION:  Science Fiction

MOVIE:  Patema Inverted 	PREDICTION:  Documentary,Music


正如上面所看到的，结果看起来很有希望，但是我们如何真正比较这两个模型呢？我们需要量化我们的表现，以便我们可以说哪一个更好。让我们回到我们刚开始讨论的内容.——我们正在学习一个函数g，它可以近似原始未知函数f。对于XI的一些值，预测肯定是错误的，我们想要最小化它。

对于多标签系统，我们经常使用“精准度”和“召回率”来评估性能。这些是标准度量，如果您对这些术语不熟悉，可以通过google阅读更多关于它们的信息。


评估矩阵
===================

我们将使用标准的 精准度-召回率 矩阵来评估我们的系统。

::
  
  def precision_recall(gt,preds):
    TP=0
    FP=0
    FN=0
    for t in gt:
        if t in preds:
            TP+=1
        else:
            FN+=1
    for p in preds:
        if p not in gt:
            FP+=1
    if TP+FP==0:
        precision=0
    else:
        precision=TP/float(TP+FP)
    if TP+FN==0:
        recall=0
    else:
        recall=TP/float(TP+FN)
    return precision,recall

::
  
  precs=[]
  recs=[]
  for i in range(len(test_movies)):
      if i%1==0:
          pos=test_movies[i]
          test_movie=movies_with_overviews[pos]
          gtids=test_movie['genre_ids']
          gt=[]
          for g in gtids:
              g_name=Genre_ID_to_name[g]
              gt.append(g_name)
            #print predictions[i],movies_with_overviews[i]['title'],gt
          a,b=precision_recall(gt,predictions[i])
          precs.append(a)
          recs.append(b)

  print np.mean(np.asarray(precs)),np.mean(np.asarray(recs))

0.33085149314 0.570960451977

::
  
  precs=[]
  recs=[]
  for i in range(len(test_movies)):
      if i%1==0:
          pos=test_movies[i]
          test_movie=movies_with_overviews[pos]
          gtids=test_movie['genre_ids']
          gt=[]
          for g in gtids:
              g_name=Genre_ID_to_name[g]
              gt.append(g_name)
            #print predictions[i],movies_with_overviews[i]['title'],gt
          a,b=precision_recall(gt,predictionsnb[i])
          precs.append(a)
          recs.append(b)

  print np.mean(np.asarray(precs)),np.mean(np.asarray(recs))

0.48893866021 0.549604519774

我们样本的平均精准度和召回率评分相当不错！模型似乎起作用了！另外，我们可以看到朴素叶贝斯的性能优于支持向量机。我强烈建议你阅读一下Multinomial叶贝斯，思考一下它为和非常适用于“文档分类”问题，这与我们的问题十分相似，因为每部电影的概览都可以被看成是需要我们分配标签的文档。

