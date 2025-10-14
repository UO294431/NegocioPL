import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from scipy.spatial import distance

Square=numpy.meshgrid(numpy.linspace(-1.1,1.1,512),numpy.linspace(-1.1,1.1,512),indexing='ij')
X=Square[0]; Y=Square[1]

f=lambda x,y,p: minkowski([x,y],[0.0,0.0],p)<=1.0
eu = lambda x,y: distance.euclidean([x,y],[0.0,0.0])<=1.0
ch = lambda x,y: distance.chebyshev([x,y],[0.0,0.0])<=1.0
ci = lambda x,y: distance.cityblock([x,y],[0.0,0.0])<=1.0

Ball=lambda p:numpy.vectorize(f)(X,Y,p)
BallEu= numpy.vectorize(eu)(X,Y)
BallCh= numpy.vectorize(ch)(X,Y)
BallCi= numpy.vectorize(ci)(X,Y)


# plt.imshow(Ball(1)); plt.axis('off'); plt.show()
# plt.imshow(Ball(2)); plt.axis('off'); plt.show()
# plt.imshow(Ball(3)); plt.axis('off'); plt.show()
# plt.imshow(Ball(4)); plt.axis('off'); plt.show()

plt.imshow(BallEu); plt.axis('off'); plt.show()
plt.imshow(BallCh); plt.axis('off'); plt.show()
plt.imshow(BallCi); plt.axis('off'); plt.show()
