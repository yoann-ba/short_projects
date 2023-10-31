# Dynamic mosaic (semi-WIP)

Project to break an image into a mosaic, except the polygon sizes vary over the image in an attempt to have large polygons over areas of low color variations, and small ones over area of high color detail.

So far only using a grid of points that is then used as the base for a Voronoi plot, and all the optimising/modifying happens in the point grid placement.

Also a way to test out the OkLab color space. https://bottosson.github.io/posts/oklab/

- method 1 (in *_sandbox file): compute a gradient of color variation (worked ok-ish), then use it to spawn points around the gradient lines. Didn't work well enough as an entire pipeline

- method 2 (in *_merge file): spawn points everywhere, then iteratively merge some that are too close and/or too similar in color. Works much better than method 1. But can be somewhat slow.

Points are spawned randomly, then a KDTree is built, then every point looks at the color difference in Lab space with tis neighbour and under some threshold they are merged into their middle point. This iterate until a satisfying grid is reached, eventually though this converges to one solution that stops changing over iterations, but it may not be the final iteration that is the best visually. This is not implemented i sub pixel way, although resizing the image virtually does that anyway.

There are still ways to change/optimize this.