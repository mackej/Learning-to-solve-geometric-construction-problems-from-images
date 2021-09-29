import numpy
import cairo
scale = 1
width, height = 255, 255
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
cr = cairo.Context(surface)

cr.set_source_rgb(1, 1, 1)
cr.rectangle(0, 0, width * scale, height * scale)
cr.fill()
buf = surface.get_data()
data = numpy.ndarray(shape=(width, height, 4), dtype=numpy.uint8, buffer=buf)
print(data)
