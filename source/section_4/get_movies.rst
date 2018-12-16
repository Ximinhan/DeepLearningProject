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

  # This function just generates all possible pairs of movies
  def list2pairs(l):
      # itertools.combinations(l,2) makes all pairs of length 2 from list l.
      pairs = list(itertools.combinations(l, 2))
      # then the one item pairs, as duplicate pairs aren't accounted for by itertools
      for i in l:
          pairs.append([i,i])
      return pairs

深入探讨同时出现的类型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
