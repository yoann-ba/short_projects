# Painting by Numbers

[YT video](https://youtu.be/1JlmZMAZ7G0).

Gave myself the challenge of looking into simulating Painting by Numbers in 5 hours. Here is what I found, the quick/light methods I tested, and the end results. Feel free to just watch the results at the end, the rest is tech details.

In the end, it's not perfect as there remains a lot of little tiny cells and too many similar-looking colors are often chosen in the palette, which would respectively need some work on a post-processing operation and on re-weighting the KMeans calculation to favor small clusters more. I still think some interesting-looking images can be generated from this, especially considering it's a 1-2 seconds single cpu core computation on a 1920*1080 image, and just image-wide operations. Could also look into replacing the bilinear filter with the generalised/anisotropic kuwahara filter.

First video of this kind, did just one audio take without over-editing it so I could fully see the result, so it's quite bad. Going from explaining something in french to someone in person is really not quite the same as doing it in english to my fake desk plant...

Chapters : 
```
0:00 Start
0:10 Intro
0:26 Painting by numbers
0:44 Quick research
1:18 CIELab Color Space (Perceptual Uniformity)
2:50 First approach (Downsample + Top n color frequency)
4:25 Second approach (Blur + KMeans + Sharpen)
8:00 Summary of the approach
8:24 One detailed example
8:48 Bad examples
9:43 Good examples
```

Showcase music : 
Synthesia by Dakooters

(for some reason I'm not allowed to make the links clickable rip)
Links : 
- Quick Research
  - Example of a website where you can load your own image https://pbnify.com/
  - Stack Exchange thread https://codegolf.stackexchange.com/questions/42217/paint-by-numbers
- CIELab Color Space
  - OpenCV Color Space Conversion https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
  - CIELab Color Space https://en.wikipedia.org/wiki/CIELAB_color_space
  - For better perceptual uniformity, see Color Difference (Delta E*) https://en.wikipedia.org/wiki/Color_difference
  - In general, for color spaces, https://poynton.ca/Poynton-color.html
- Second approach
  - KMeans https:// en.wikipedia.org/wiki/K-means_clustering https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
(which is essentially just a Voronoi-Lloyd https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)

Data/Code availability : 
soon(tm)
