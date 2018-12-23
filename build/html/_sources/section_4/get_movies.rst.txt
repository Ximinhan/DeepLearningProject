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

我们现在要做的是寻找一些同时出现类型的最佳集合，并看看逻辑上对我们是否有用？直观来说，看到上图呈现出正方形聚集不会太意外--大多数分布在正方形区域也就是说某些类型总是同时出现而与其他类型却很少交互。某种意义上说，这将凸显并区分出哪些类型同时出现而哪些不是。


尽管这些数据也许不会直接显示，但我们可以分析一下这些数据。这里用到的技术叫做双聚类分析。

::

  from sklearn.cluster import SpectralCoclustering
  model = SpectralCoclustering(n_clusters=5)
  model.fit(visGrid)

  fit_data = visGrid[np.argsort(model.row_labels_)]
  fit_data = fit_data[:, np.argsort(model.column_labels_)]

  annot_lookup_sorted = []
  for i in np.argsort(model.row_labels_):
      annot_lookup_sorted.append(Genre_ID_to_name[nr_ids[i]])
    
  sns.heatmap(fit_data, xticklabels=annot_lookup_sorted, yticklabels=annot_lookup_sorted, annot=False)
  plt.title("After biclustering; rearranged to show biclusters")

  plt.show()

.. image:: bicluster.png

请看上图，“正方形”或者说一组电影类型自动出现了！

直观来看，犯罪，科幻，悬疑，动作，戏剧，惊悚等类型同时出现，另一方面，浪漫，科幻，家庭，音乐，冒险等类型同事出现。

这很有直观意义，对吧？

一个比较有挑战性的地方是戏剧类型具有广泛分布。它使得图中两大类集合高度重叠。如果我们使其与动作，惊悚等类型合并，那么几乎所有电影都会贴上这个标签。

根据以上内容的分析，我们可以将数据归类为“戏剧、动作、科幻、刺激（惊悚、犯罪、神秘）、振奋（冒险、幻想、动画、喜剧、浪漫、家庭）、恐怖、历史”等流派。

注意：这种分类是主观的，绝不是唯一的正确解决方案。也可以只保留原始标签，只排除那些没有足够数据的标签。这些技巧对于平衡数据集很重要，它允许我们增加或减少某些信号的强度，使得改进我们的推断成为可能：）


有趣的问题
~~~~~~~~~~~~~~~~~~

在这你可以尽情发挥想象力，应该会想出一些比我更好的问题。

以下是一些我的想法：
 - 在一些电影中，哪些演员总是演某种类型，而有些演员却不拘泥于特定类型？
 - 在最近几年中是否有某种流行趋势？
 - 你能用音轨来识别电影的类型吗？
 - 著名的浪漫类型演员的薪水是否比著名的动作演员高？
 - 如果你看上映日期和流行度得分，哪种电影类型有更长的保质期？

关于按种类拓展特征相关性的一些想法：
 - 标题的长度是否与电影类型有关？
 - 恐怖片相较于浪漫和喜剧片，电影海报是否颜色更阴暗？
 - 某些类型电影是否在一年中某些特殊档期发布更多？
 - RPG等级与电影类型有关么？


基于这个新的数据集，我们现在可以从TMDB抓取海报来作为我们的训练数据了！
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::
  
  # 在开始之前，我们从pickle文件中读取数据以保持数据一致性！
  # 我们现在对每个类型采样100部电影。有一个问题是由于排名是按流行程度排序的，因此可能会有重复。
  # 我们需要移除已经采样的那些电影。
  movies = []
  baseyear = 2017

  print('Starting pulling movies from TMDB. If you want to debug, uncomment the print command. This will take a while, please wait...')
  done_ids=[]
  for g_id in nr_ids:
      #print('Pulling movies for genre ID '+g_id)
      baseyear -= 1
      for page in xrange(1,6,1):
          time.sleep(0.5)
    
          url = 'https://api.themoviedb.org/3/discover/movie?api_key=' + api_key
          url += '&language=en-US&sort_by=popularity.desc&year=' + str(baseyear) 
          url += '&with_genres=' + str(g_id) + '&page=' + str(page)

          data = urllib2.urlopen(url).read()

          dataDict = json.loads(data)
          movies.extend(dataDict["results"])
      done_ids.append(str(g_id))
  print("Pulled movies for genres - "+','.join(done_ids))
  #Pulled movies for genres - 12

现在开始从TMDB获取电影数据。如果你想调试可以将那条打印语句取消注释。这会运行一段时间，请耐心等待。。。

::
  
  f6=open("movies_for_posters",'rb')
  movies=pickle.load(f6)
  f6.close()

让我们去除列表中那些重复数据。

::
  
  movie_ids = [m['id'] for m in movies]
  print "originally we had ",len(movie_ids)," movies"
  movie_ids=np.unique(movie_ids)
  print len(movie_ids)
  seen_before=[]
  no_duplicate_movies=[]
  for i in range(len(movies)):
      movie=movies[i]
      id=movie['id']
      if id in seen_before:
          continue
  #         print "Seen before"
      else:
          seen_before.append(id)
          no_duplicate_movies.append(movie)
  print "After removing duplicates we have ",len(no_duplicate_movies), " movies"

originally we had  1670  movies
1608
After removing duplicates we have  1608  movies

同时，我们删除那些没有海报的电影！

::

  poster_movies=[]
  counter=0
  movies_no_poster=[]
  print("Total movies : ",len(movies))
  print("Started downloading posters...")
  for movie in movies:
      id=movie['id']
      title=movie['title']
      if counter==1:
          print('Downloaded first. Code is working fine. Please wait, this will take quite some time...')
      if counter%300==0 and counter!=0:
          print "Done with ",counter," movies!"
          print "Trying to get poster for ",title
      try:
          grab_poster_tmdb(title)
          poster_movies.append(movie)
      except:
          try:
              time.sleep(7)
              grab_poster_tmdb(title)
              poster_movies.append(movie)
          except:
              movies_no_poster.append(movie)
      counter+=1
  print("Done with all the posters!")

('Total movies : ', 1670)
Started downloading posters...
Downloaded first. Code is working fine. Please wait, this will take quite some time...
Done with  300  movies!
Trying to get poster for  Gravity
Done with  600  movies!
Trying to get poster for  Zombieland
Done with  900  movies!
Trying to get poster for  The Substitute
Done with  1200  movies!
Trying to get poster for  Decoys
Done with  1500  movies!
Trying to get poster for  Lost and Delirious
Done with all the posters!

::

  print len(movies_no_poster)
  print len(poster_movies)

170
1500

::
  
  f=open('poster_movies.pckl','r')
  poster_movies=pickle.load(f)
  f.close()

::
  
  f=open('no_poster_movies.pckl','r')
  movies_no_poster=pickle.load(f)
  f.close()

恭喜，我们已经完成了信息爬取！

基于离散信息建立数据集！
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

这个任务很简单但是非常重要。基本上这将构建整个项目的基础。鉴于你可以在我提供的框架中自由构建自己的项目，你必须做出许多决定来完成你自己版本的项目。

当我们处理分类问题时，我们需要根据手头的数据做出两个决定：

 - 我们想要预测什么，也就是我们的 Y 是什么？
 - 我们用什么特征去预测这个 Y ，也就是我们应该用什么作为 X ？

有很多不同的选项，由你来决定什么是最好的。我将以我的版本为例，但你必须思考一下然后想出一个你认为最好的版本！

作为举例，这里有很多方式构建 Y，仍然拿类型预测问题来说：

 - 假设每个电影都有多种类型，这就变成了一个多标签分类问题。比方说，一个电影可以同时是动作类，恐怖类和探险类。也就是说，每部电影可以有多个分类。
 - 利用我们在第一节学到的双聚类分析生成类型组，这样每个电影就只有一个类型了。这样，问题就简化成了多类问题。比方说某个电影拥有这样的类型-刺激的，或是恐怖的，或是历史的。每个电影就只有一个分类。

为了实现这个目的，我将从上面的第一种情况开始-即多标签分类问题。

类似地，为了设计我们的输入特性，即X，您可以选择您认为有意义的任何特性，例如，电影的导演可能是流派的良好预测器。或者，他们可以选择使用PCA等算法设计的任何特性。鉴于IMDB、TMDB和维基百科等替代资源的丰富性，有很多可用的选项。这里要有创意！

另一件需要注意的重要事情是，在这样做时，我们还必须在途中做出许多更小的实现决策。例如，我们将包括哪些流派？我们要包括哪些电影？所有这些都是开放式的！

我的实现
~~~~~~~~~~~~~~~~~

实现步骤-

 - 如上所述问题被简化为一个多标签问题
 - 我们将尝试预测一部电影关联的多种类型，这将作为我们的Y
 - 我们使用两个不同的类型作为 X -- 文本和图片
 - 对文本部分--被用来预测类型的输入特征使用的是电影情节，可以从TMDB获取的数据中'overview'属性得到。这将作为我们的X
 - 对图片部分--我们使用爬取的电影海报图片作为我们的X

注意：我们将首先看一些传统的机器学习模型，这些模型在最近兴起的神经网络和深度学习之前很流行。对电影海报图像进行类型预测，我之所以避免使用这种方法是因为在不使用深度学习的情况下，传统的机器学习模型不再被用来特征提取（前面都详细讨论过，不要害怕术语）。对于用电影概述来预测电影类型的问题，我们将同时使用传统模型和深度学习模型。

现在，我们开始建立我们的 X 和 Y ！

首先，我们筛选出那些有概述的电影。接下来的步骤是用来解释为什么数据清洗很重要的很好的案例！

::

  movies_with_overviews=[]
  for i in range(len(no_duplicate_movies)):
      movie=no_duplicate_movies[i]
      id=movie['id']
      overview=movie['overview']
    
      if len(overview)==0:
          continue
      else:
          movies_with_overviews.append(movie)
        
  len(movies_with_overviews)

1595

现在，我们把这些电影的类型存储在列表中，后面我们会把他们转化为布尔向量。

布尔向量表示法是机器学习中存储/表示数据的一种常见且重要的方法。本质上，它是将具有n个可能值的分类变量简化为n个布尔变量的一种方法。什么意思呢？比方说，[(1,3),(4)]这个列表表示样本A有两个标签1和3,而样本B有一个标签4。对每个样本来说，对于每个可能的标签，表示法简化为如果它具有该标签那么用1表示，如果没有该标签那么用0表示。上述列表的布尔表示法版本是-

[(1,0,1,0),(0,0,0,1)]

::

  # genres=np.zeros((len(top1000_movies),3))
  genres=[]
  all_ids=[]
  for i in range(len(movies_with_overviews)):
      movie=movies_with_overviews[i]
      id=movie['id']
      genre_ids=movie['genre_ids']
      genres.append(genre_ids)
      all_ids.extend(genre_ids)

::
  
  from sklearn.preprocessing import MultiLabelBinarizer
  mlb=MultiLabelBinarizer()
  Y=mlb.fit_transform(genres)

::

  genres[1]

[28, 12, 35, 10749]

::

  print Y.shape
  print np.sum(Y, axis=0)

(1595, 20)
[327 234 220 645 222 438 404 138 45 440 233 133 242 196 135 232 256 80 23 25]

::
  
  len(list_of_genres)

19

这很有趣，如果你还记得的话，我们只有19个类型标签。但是得出 Y 的结果是 1666,20 ,而它应该是 1666,19 ，因为我们只有19个种类？我们来看看怎么回事。

让我们找出那些不属于我们原始类型列表中的类型ID！

::

  # Create a tmdb genre object!
  genres=tmdb.Genres()
  # the list() method of the Genres() class returns a listing of all genres in the form of a dictionary.
  list_of_genres=genres.list()['genres']
  Genre_ID_to_name={}
  for i in range(len(list_of_genres)):
      genre_id=list_of_genres[i]['id']
      genre_name=list_of_genres[i]['name']
      Genre_ID_to_name[genre_id]=genre_name

::
  
  for i in set(all_ids):
    if i not in Genre_ID_to_name.keys():
        print i

10769

这个类型ID并不是我们查询TMDB获取所有可能的类型得到的。我们现在怎么办？我们不能忽略所有包含此类型的样本。但是如果你向前看你会看到许多这样的样本。所以我google了一下又看了他们的文档发现这个ID对应的类型是'Foreign'。因此我们把它加入到我们自己定义的字典中。像这样的问题在机器学习中很常见，当我们遇到时需要调查并纠正过来。我们总是要决定哪些要保留，如何保存等等。

::
  
  Genre_ID_to_name[10769]="Foreign" #Adding it to the dictionary
  len(Genre_ID_to_name.keys())

20

现在我们来建立 X 矩阵也就是输入特征！之前说过，我们会使用电影概述作为我们的输入向量！我们来看一个例子！

::
 
  sample_movie=movies_with_overviews[5]
  sample_overview=sample_movie['overview']
  sample_title=sample_movie['title']
  print "The overview for the movie",sample_title," is - \n\n"
  print sample_overview

The overview for the movie Doctor Strange  is - 


After his career is destroyed, a brilliant but arrogant surgeon gets a new lease on life when a sorcerer takes him under his wing and trains him to defend the world against evil.

我们如何在矩阵中存储电影概述？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
