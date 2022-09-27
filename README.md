# H-QAM
## HQAM Constellations Detection 

Paper [1] proposes a simple detection method for hexagonal-qam constellations. This is an attempt for a faster but as efficient method.

In this project we propose a new detection method based on an adaptation of code from [2]
and with the help of [3]. 

### Constallation formation: Regular-Irregular HQAM

### Detection: 

The received signal can be expressed as a point in cartesian coordinates (x,y). As the decision areas for hqam can be approximated with hexagons, we transform our coordinates into hexagonal coordinates. Then, with basic maths we can derive in which hexagon lies our received signal. Some simulations showing that our detection method follows the analysis proposed in [1]:

![Thrass approx](https://user-images.githubusercontent.com/70851911/192646814-96dbc256-2749-4a2f-b975-7817d89f1760.png)

With this method we achieve O(1) complexity as the detection uses only two round() and 3 ceil()

[1] "On the Error Analysis of Hexagonal-QAM
Constellations" Thrassos K. Oikonomou, Student Member, IEEE, Sotiris A. Tegos, Student Member, IEEE,
Dimitrios Tyrovolas, Student Member, IEEE, Panagiotis D. Diamantoulakis, Senior Member, IEEE,
and George K. Karagiannidis, Fellow, IEEE 
[2] https://www.redblobgames.com/grids/hexagons

[3]https://justinpombrio.net/programming/2020/04/28/pixel-to-hex.html
