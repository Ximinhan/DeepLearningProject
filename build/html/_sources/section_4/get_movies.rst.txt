构建要使用的数据集:让我们看看数据库中的前1000部电影
==========================================================

使用前面的API我们只获取前50个页面。如前文所说，可以使用 top_movies=all_movies.popular() 的"page"参数实现。

注意：下面的一些代码会在名为 "pickle" 的python文件中存储一些数据用来被内存访问，而不是每次都重新下载。一旦完成，你应该注释掉这些会产生新对象的代码并不再使用。

::

  all_movies=tmdb.Movies()
  top_movies=all_movies.popular()

  len(top_movies['results'])
  top20_movs=top_movies['results']

::

  # 当数据保存到pickle文件后请注释掉这段代码
  all_movies=tmdb.Movies()
  top1000_movies=[]
  print('Pulling movie list, Please wait...')
  for i in range(1,51):
      if i%15==0:
          time.sleep(7)
      movies_on_this_page=all_movies.popular(page=i)['results']
      top1000_movies.extend(movies_on_this_page)
  len(top1000_movies)
  f3=open('movie_list.pckl','wb')
  pickle.dump(top1000_movies,f3)
  f3.close()
  print('Done!')

::
  
  f3=open('movie_list.pckl','rb')
  top1000_movies=pickle.load(f3)
  f3.close()

结对分析电影类型
~~~~~~~~~~~~~~~~~~~~~~~~~~~

由于我们的数据集是多标签的，仅仅只是关注类型分类是不充分的。观察哪些类型同时出现可能是有益的，因为它可能有助于揭示我们数据集中的固有偏见。例如，如果浪漫和喜剧比纪录片和喜剧更经常地出现在一起，那是有意义的。这种固有的偏见告诉我们，我们从自身抽样的潜在人群是不平衡的。然后我们可以采取步骤来考虑这些问题。即使我们不采取这些步骤，重要的是要意识到我们所做的假设是不平衡的数据集不会损害我们的性能，如果需要，我们可以回过头来修正这个假设。古老但有用的科学方法，不是吗?

所以对于前1000部电影，我们来做一些类型分布的结对分析。我们的主要目的是看哪些类型的电影出现在同一部电影中。因此，我们首先定义一个函数，它取一个列表，并从中生成所有可能的对。然后，我们提取电影的类型列表，并在类型列表上运行这个函数，以获得所有同时出现的类型对。

::

  # 此方法生成所有可能的电影组合
  def list2pairs(l):
      # itertools.combinations(l,2) 从列表l中生成长度为2的组合
      pairs = list(itertools.combinations(l, 2))
      # 然后是长度为1的组合，重复的组合不会被itertools计算
      for i in l:
          pairs.append([i,i])
      return pairs

如我所说，现在我们要抓取每部电影的类型数据，然后使用上面的方法计算两种类型同时出现的概率

::
  
  # get all genre lists pairs from all movies
  allPairs = []
  for movie in top1000_movies:
      allPairs.extend(list2pairs(movie['genre_ids']))
    
  nr_ids = np.unique(allPairs)
  visGrid = np.zeros((len(nr_ids), len(nr_ids)))
  for p in allPairs:
      visGrid[np.argwhere(nr_ids==p[0]), np.argwhere(nr_ids==p[1])]+=1
      if p[1] != p[0]:
          visGrid[np.argwhere(nr_ids==p[1]), np.argwhere(nr_ids==p[0])]+=1

让我们看看刚刚新建的数据结构。这是一个如下所示的19X19的结构。也就是说我们有19中类型，不用说，这种结构计算了同一部电影中同时出现的类型的数量。

::
  
  print visGrid.shape
  print len(Genre_ID_to_name.keys())

  (19, 19)
  19

::

  annot_lookup = []
  for i in xrange(len(nr_ids)):
      annot_lookup.append(Genre_ID_to_name[nr_ids[i]])

  sns.heatmap(visGrid, xticklabels=annot_lookup, yticklabels=annot_lookup)

.. image:: 19genres.png

上图以热力图的形式展现了经常同时出现的类型的分布

在上面的图中需要注意的重要一点是对角线。对角线对应的是自我对应，即一个类型出现的次数，例如戏剧与戏剧一起出现的次数。这基本上可以认为是一种类型发生的总次数的计数!

我们也可以看到许多数据集中在戏剧类型上，当然这也是一个很通用的类型标签。几乎没有纪录片或者是电视电影出现。恐怖片是一个非常独特的类别，浪漫类型也不是分布很广。

为了了解这些数据不均匀的原因，我们可以尝试多种方式来探索可能发现的一些有趣关系。


深入探索同时出现的类型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
