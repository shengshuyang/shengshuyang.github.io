---
layout: post
title:  "Binary Representation"
date:   2016-05-18 00:45:21 -0700
categories: Coding
---

### The Question 

Given a (decimal - e.g. 3.72) number that is passed in as a string, return the binary representation that is passed in as a string. If the fractional part of the number can not be represented accurately in binary with at most 32 characters, return ERROR.

Example

{% highlight html %}
For n = "3.72", return "ERROR".
For n = "3.5", return "11.1".
{% endhighlight %}

### The Caveate

The problem itself is not difficult, but there are many corner cases to consider, such as *0.1*,  *1.0*, *1* , etc., as well as the case where the fractal part cannot be represented by 32 bit integer.

### The Solution

{% highlight python %}
class Solution:
    #@param n: Given a decimal number that is passed in as a string
    #@return: A string
    def binaryRepresentation(self, n):
        # write you code here
        
        # create two strings that are used as cache
        b_intg = ""
        b_frac = ""
        
        # if there's no decimal point, we can't split the string with dot.
        if '.' in n:
            [intg, frac] = n.split('.')
            intg, frac = int(intg), float("0."+frac)
        else:
            intg = int(n)
        
        # construct the integer part
        temp = intg
        while temp > 0:
            b_intg = str(temp%2) + b_intg
            temp = temp / 2
        
        # construct the fractal part
        temp = frac
        count = 0
        while frac > 0:
            count += 1
            if count >= 32:
                return "ERROR"
            frac = frac * 2
            b_frac =  b_frac + str(frac)[0]
            if frac >= 1:
                frac = frac - 1
        
        # if b_intg is empty, we know that intg is zero, so we need to specify
        # that by pushing a leading zero here.
        if b_intg == "":
            b_intg = "0"
        # if b_frac is empty, we simply return the integer part, no decimal point.
        if b_frac == "":
            return b_intg
        # if both are non-empty, then we join the two parts with a decimal point.
        return '.'.join([b_intg, b_frac])
{% endhighlight %}
            
