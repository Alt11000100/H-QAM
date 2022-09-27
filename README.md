# H-QAM
HQAM Constellations Detection 

Paper [1] proposes a simple detection method for hexagonal-qam constellations. This is an attempt for a faster but as efficient method.

In this project we propose a new detection method based on an adaptation of code from #https://www.redblobgames.com/grids/hexagons/#neighbors   [2]
and with the help of https://justinpombrio.net/programming/2020/04/28/pixel-to-hex.html    [3]. 

-Constallation formation: Regular-Irregular HQAM
-Detection: 

The received signal can be expressed as a point in cartesian coordinates (x,y). As the decision areas for hqam are hexagons we transform our coordinates into hexagonal coordinates. Then with basic maths we can derive in which hexagon lies our received signal. Some simulations showing that our detection method follows the analysis proposed in [1]:

