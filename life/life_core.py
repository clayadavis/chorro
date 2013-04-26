import numpy
import scipy.ndimage
import Image

class Life(object):

    def __init__(self, n, p=0.5, mode='wrap'):
        self.n = n
        self.mode = mode
        self.array = numpy.uint8(numpy.random.random((n, n)) < p)
        self.weights = numpy.array([[1,1,1],
                                    [1,10,1],
                                    [1,1,1]], dtype=numpy.uint8)

    def step(self):
        con = scipy.ndimage.filters.convolve(self.array,
                                             self.weights,
                                             mode=self.mode)

        boolean = (con==3) | (con==12) | (con==13)
        self.array = numpy.int8(boolean)
        
    def run(self, N):
        for _ in range(N):
            self.step()
        
    def draw(self, scale):
        im = Image.fromarray(numpy.uint8(self.array)*255)
        z = int(scale*self.n)
        return im.resize((z,z))
