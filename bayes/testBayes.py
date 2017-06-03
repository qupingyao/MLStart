#-*- coding: UTF-8 -*-
from bayes import *
from feedparser import *

testingNB()
ny = parse("http://newyork.craigslist.org/stp/index.rss")
sf = parse("http://sfbay.craigslist.org/stp/index.rss")
vocabList,p0V,p1V = localWords(ny,sf)
getTopWords(ny,sf)
