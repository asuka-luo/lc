# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math

def judgeSquareSum(c):
    a_max = int(math.sqrt(c))
    for a in range(a_max, -1, -1):
        b_square = c - a**2
        print("a",a)
        print("b_square", b_square)
        b_possible = int(math.sqrt(b_square))
        if int(b_square) == int(b_possible**2):
            return True
    return False

import itertools
def combinationSum(candidates, target):
    def dfs(begin, path,  target):
        if target < 0 :
            return 
        if target == 0:
            res.append(path)
            return
        for index in range(begin, size):
            dfs( index, path+[candidates[index]], target- candidates[index])
    size = len(candidates)
    if size == 0:
        return []
    path = []
    res = []
    dfs(0,  path, target)
    return res


def leastBricks(wall):
    interval = []
    for each_list in wall:
        interval.append([sum(each_list[:x]) for x in range(1, len(each_list)+1)])
    total = [x for y in interval for x in y]
    total = list(set(total))
    if len(total) == 1:
        return len(wall)
    cur_max = 0
    for x in total[:-1]:
        counter = 0
        for y in interval:
            if x in y:
                counter += 1
        if counter > cur_max:
            cur_max = counter
    return len(wall) - cur_max

import collections
def leastBricks_2( wall):

    width = sum(wall[0])
    height = len(wall)
    counts = collections.defaultdict(lambda: height)
    for row in wall:
        _sum = 0 
        for r in row[:-1]:
            _sum += r
            counts[_sum] -= 1
        if len(counts.values()):
            return min(counts.values())
        else:
            return height
        
def minimumTimeRequired(jobs, k):
    from heapq import heapify, heappush, heappop
    heap = [0] * k
    heapify(heap)
    jobs =  sorted(jobs)[::-1]
    for i in jobs:
        heappush(heap, heappop(heap) + i)
    m = max(heap)
    
    a = [0] * k
    def job(j):
        nonlocal m 
        if j == len(jobs):
            m = min(m, max(a))
            return 
        for i in range(min(k, j+1)):
            if a[i] + jobs[j] > m:
                continue
            a[i] += jobs[j]
            job(j+1)
            a[i] -= jobs[j]
    job(0)
    return m

        
#这道题是leetcode 1269
#其实这道题用dp的方法更好做，运行方法更快，具体放在下方
from functools import lru_cache
def numWays( steps, arrLen):
    @lru_cache(None)
    def dfs(ind, step_num):
        if step_num == steps:
            if ind == 0:
                res[0] += 1
                return
            else:
                return 
        if ind == 0:
            for x in range(2):
                dfs(ind+x, step_num +1)
        elif ind == arrLen - 1:
            for x in range(-1, 1):
                dfs(ind+x, step_num + 1)
        else:
            for x in range(-1, 2):
                dfs(ind+x, step_num +1)
    res = [0] 
    dfs(0,0)
    return res[0]


def numWays( steps, arrLen):
    arrLen = min(steps // 2 + 1, arrLen)
    dp = [1] + [0] * (arrLen - 1) 
    mod_num = 1000000007
    for _ in range(steps):
        s = [0] * arrLen
        s[0] = (dp[0] + dp[1]) % mod_num
        s[-1] = (dp[-1] + dp[-2]) % mod_num
        for x in range(1, arrLen - 1):
            s[x] = (dp[x-1] + dp[x] + dp[x+1]) % mod_num
        dp = s
    return dp[0]

class Solution():
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        ans = []
        def dfs(cur_li, to_choose, tar):
            print(cur_li)
            if tar == 0:
                if cur_li not in ans:
                    ans.append(cur_li)
                return
            elif tar < 0:
                return 
            for ind,x in enumerate(to_choose):
                temp = to_choose[ind+1:]
                dfs(cur_li + [x], temp, tar - x)
        dfs([], candidates, target)
        return ans
    
class Solution():
    def numMatchingSubseq(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: int
        """
        waiting = defaultdict(list)
        for w in words:
            waiting[w[0]].append(iter(w[1:]))
        for c in s:
            for it in waiting.pop(c,()):
                print(it)
                waiting[next(it, None)].append(it)
        return len(waiting[None])