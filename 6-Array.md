# 6-Array
## 54. Spiral Matrix (Medium)
https://leetcode.com/problems/spiral-matrix/
*Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
有可能不是square matrix*
```python
class Solution(object):
    def spiralOrder(self, matrix):
        if not matrix: return []
        
        res = []
        rBegin = 0
        rEnd = len(matrix)-1    # row边界
        cBegin = 0
        cEnd = len(matrix[0])-1 # colum边界
        
        while rBegin <= rEnd and cBegin <= cEnd:
            # from left to right
            for j in range(cBegin, cEnd +1):
                res.append(matrix[rBegin][j])
            rBegin += 1 # 删掉顶端一行
            
            # traverse down
            for j in range(rBegin, rEnd+1):
                res.append(matrix[j][cEnd])
            cEnd -= 1
            
            # from right to left
            if rBegin <= rEnd:  # 还有行未遍历
                for j in range(cEnd, cBegin - 1, -1):
                    res.append(matrix[rEnd][j])
            rEnd -= 1
            
            # traverse up
            if cBegin <= cEnd:  # 还有列未遍历
                for j in range(rEnd, rBegin - 1, -1):
                    res.append(matrix[j][cBegin])
            cBegin += 1
            
        return res
```
## 59. Spiral Matrix II
*Input: 3
Output:
[
 [ 1, 2, 3 ],
 [ 8, 9, 4 ],
 [ 7, 6, 5 ]
]
一定是square matrix*

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for i in range(n)]
        rBegin = 0
        rEnd = n - 1
        cBegin = 0
        cEnd = n - 1
        num = 1
        while rBegin <= rBegin and cBegin <= cEnd:
            for j in range(cBegin, cEnd + 1):
                matrix[rBegin][j] = num
                num += 1
            rBegin += 1
            
            for j in range(rBegin, rEnd + 1):
                matrix[j][cEnd] = num
                num += 1
            cEnd -= 1
            
            for j in range(cEnd, cBegin - 1, -1):
                matrix[rEnd][j] = num
                num += 1
            rEnd -= 1
            
            for j in range(rEnd, rBegin - 1, -1):
                matrix[j][cBegin] = num
                num += 1
            cBegin += 1
        return matrix
```
==ATTENTION!!!==
对矩阵的定义：
Wrong
```python
n = 3
m =  [[0] * n] * n
m[1][1] = 9
print(m)
>>> [[0, 9, 0], [0, 9, 0], [0, 9, 0]]
```
Correct
` [[0] * n] * n` 这种定义，每一行都是一样的复制，一行的值改变，所有行一起改变

```python
n = 3
m = [[0] * n for i in range(n)]
m[1][1] = 9
print(m)
>>> [[0, 0, 0], [0, 9, 0], [0, 0, 0]]
```
`[[0] * n for i in range(n)]` 这是我们需要的定义，每行独立
```python
m = [[0 for _ in range(n)] for i in range(n)]
```
这个与上一个方法相同。

## 42. Trapping Rain Water (Hard)
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

![在这里插入图片描述](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9hc3NldHMubGVldGNvZGUuY29tL3VwbG9hZHMvMjAxOC8xMC8yMi9yYWlud2F0ZXJ0cmFwLnBuZw?x-oss-process=image/format,png#pic_center)
The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. 

1. Brute force
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200906024719185.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0hhbmhhaGFoYWhhaA==,size_16,color_FFFFFF,t_70#pic_center)
```python
class Solution:
    # brute force
    def trap(self, height: List[int]) -> int:
        ans = 0
        for i, h in enumerate(height):
            lmax = max(height[:i+1])    #  要包含i自己，因为有可能当前i是最高的，那么curr-h=0
            rmax = max(height[i:])
            curr = min(lmax, rmax) - h
            ans += curr
            
        return ans
```
做了很多的重复扫描左右的最大值。

2. DP
用空间换时间，将最大值储存在数组中。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200906052425268.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0hhbmhhaGFoYWhhaA==,size_16,color_FFFFFF,t_70#pic_center)
```python
class Solution:
    # DP
    def trap(self, height: List[int]) -> int:
        n = len(height)
        ans = 0
        l = [0 for _ in range(n)]   # 储存i左边最高
        r = [0 for _ in range(n)]   # 储存i右边最高
        
        for i in range(0, n):
            # 防止数组越界，排除边界条件
            if i == 0:
                l[i] = height[i]
            else:
                l[i] = max(l[i-1], height[i])
                
        for i in reversed(range(n)):
            if i == n-1:
                r[i] =  height[i]
            else:
                r[i] = max(r[i+1], height[i])
            
        for i in range(n):
            curr =  min(l[i], r[i]) - height[i]
            ans += curr
            
        return ans
```
3. 双指针
用两个变量分别跟踪从左到右和从右到左的最大值。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200906023928989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0hhbmhhaGFoYWhhaA==,size_16,color_FFFFFF,t_70#pic_center)
```python
class Solution:
    # two pointers
    def trap(self, height: List[int]) -> int:
        n = len(height)
        if n == 0:
            return 0
        l = 0
        r = n - 1
        maxL = height[l]
        maxR = height[r]
        ans = 0 
        while l < r:
            # 永远先处理矮的那边
            if maxL < maxR:
                curr = maxL - height[l]
                ans += curr
                l += 1
                maxL = max(maxL, height[l])
            else:
                curr = maxR - height[r]
                ans += curr
                r -= 1
                maxR = max(maxR, height[r])
            
        return ans
```
## 53. Maximum Subarray (Easy)
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

1. DP
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSum = nums[0]    # 默认为第一个元素
        n = len(nums)
        if n == 1:
            return nums[0]
        for i in range(1, n):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
            maxSum = max(maxSum, nums[i])
        return maxSum
```
2. Greedy
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        subSum = 0  # sum of each subarray
        maxSum = nums[0]
        
        for num in nums:
            subSum += num
            if subSum > maxSum: # 取目前的最大
                maxSum = subSum # 更新maxSum
            if subSum < 0:      # if the sub sum dips into the negatives
                subSum = 0      # start a new sub array
        
        return maxSum
```

## 11. Container With Most Water (Medium)

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # 双指针
        left = 0
        right = len(height) - 1
        maxWater = 0
        
        while left < right:
            # 装水的容量只由两端中较矮的和两端距离差决定
            currMin = min(height[left], height[right])  # 两端较矮的
            maxWater = max(maxWater, currMin * (right - left))
            # 为了可能产生更大的容量，移动更矮的指针，为了有更高的bar
            if height[left] > height[right]:
                right -= 1
            else:
                left += 1
        return maxWater
```

## 15. 3Sum (Medium)
双指针
[-4 (i = 0), -1(left = i + 1), -1, 0, 1, 2 (right = len(nums) - 1) ]
i用来从左到右扫描整个数组，left和right为双指针。三个指针构成3sum

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort() # 为了后面确定指针移动方向
        for i in range(len(nums) - 2):
            if i != 0 and nums[i] == nums[i-1]: # 去重：i为同一个num
                continue
            left = i + 1
            right = len(nums) - 1
            while left < right:
                summ = nums[i] + nums[left] + nums[right]
                if summ == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    
                    # 去重，下一次指针移动又遇见了相同的num
                    while left < right and nums[left+1] == nums[left]:
                        left += 1
                    while left < right and nums[right-1] == nums[right]:
                        right -= 1
                    left += 1
                    right -= 1
                # 指针移动方向
                elif summ < 0:  # 因为sorted，如果sum<0：需要更大的数
                                # 因为left向右移动（变大），right向左移动（变小），left应该移动
                    left += 1
                elif summ > 0:
                    right -= 1
        return res
                
```
