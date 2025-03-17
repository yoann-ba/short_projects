# Perceptible Color Difference dictionary

Bit of a dumb idea, but just building a database of pairs of colors I can perceive the difference of (or not), essentially re-doing a MacAdam diagram.

Data is just R1 G1 B1 R2 G2 B2 Is_Different, 7 uint8 per rows. Done in a standard/default RGB space in python and plotted by matplotlib, but can obviously be used in other color spaces after a translation.

Feel free to use anywhere if you see a use for it.

But also look at : 
- https://en.wikipedia.org/wiki/Color_difference
- https://en.wikipedia.org/wiki/Color_difference#Tolerance
- https://bottosson.github.io/posts/oklab/

before since that might solve whatever you're trying to do better than this wheel re-invention.