import sys
import time
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from matplotlib import colormaps as cm
from numpy.typing import NDArray
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from vispy import app, scene
from vispy.scene import visuals

# from app.logic.timer import Timer

try:
    from OpenGL import GL
except ImportError:
    GL = None

matplotlib.use("agg")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def convert_to_intensity_image(complex_result: NDArray[np.int16]) -> NDArray[np.int32]:
    # timer = Timer(name="convert_to_intensity_image")
    inphase = complex_result[..., 0].astype(float)
    quadrature = complex_result[..., 1].astype(float)
    power = np.square(inphase) + np.square(quadrature)
    intensity_images = np.log10(power + 1)
    max: float = np.amax(intensity_images)
    intensity_images = intensity_images / max * 255
    intensity_images = np.roll(np.clip(intensity_images, 0, 255).astype(np.int32), 256, axis=2)
    # timer.log_time()
    return intensity_images


def norm_image(image: NDArray[np.int16]) -> NDArray[np.uint8]:
    max: np.int16 = np.amax(image)
    norm_image = image / max * 255
    return norm_image.astype(np.uint8)


# def surface_plot(image: NDArray[np.uint8]):
#     global fig, ax
#     x = np.arange(image.shape[0])
#     y = np.arange(image.shape[1])
#     x, y = np.meshgrid(x, y)
#     z = image
#     timer = Timer(name="surface_plot")
#     ax.plot_surface(x, y, z, cmap=cm["viridis"], linewidth=0, antialiased=False)
#     timer.log_time()
#     # fig.canvas.draw()
#     fig.canvas.draw()
#
#     # axbackground = fig.canvas.copy_from_bbox(ax.bbox)
#     buf: BytesIO = BytesIO()
#     fig.savefig(buf, format="jpeg")
#     fig.clear()
#     frame: memoryview[int] = buf.getbuffer()
#     return frame


class SurfacePlotUpdater:
    def __init__(self):
        """
        Initializes the SurfacePlotUpdater.
        Precomputes some parameters for optimized updates.
        """
        self.initialized = False
        self.fig = None
        self.ax = None
        self.background = None
        self.surf = None
        self.X = None
        self.Y = None
        self.last_shape = None

    def update(self, arr: NDArray[np.uint8]) -> memoryview:
        """
        Updates (or creates) the 3D surface plot using the provided 2D np.uint8 array.
        The plot is rendered using matplotlib with blitting to speed up updates.

        Parameters:
            arr (np.ndarray): A 2D numpy array of type np.uint8 used as the height values.

        Returns:
            memoryview: A memoryview of the PNG bytes of the current rendered plot.
        """
        # Verify input.
        if arr.ndim != 2 or arr.dtype != np.uint8:
            raise ValueError("Input array must be 2-dimensional and of type np.uint8")

        nrows, ncols = arr.shape
        # Precompute grid only if the shape changes.
        if self.last_shape != arr.shape:
            x = np.linspace(0, 1, ncols)
            y = np.linspace(0, 1, nrows)
            self.X, self.Y = np.meshgrid(x, y)
            self.last_shape = arr.shape

        # Create figure only once.
        if not self.initialized:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection="3d")
            # Create initial surface.
            self.surf = self.ax.plot_surface(self.X, self.Y, arr, cmap="viridis")
            # Force full draw first time to cache the background.
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.initialized = True
        else:
            # Use blitting for fast update:
            self.fig.canvas.restore_region(self.background)
            # Instead of clearing the entire axes, remove only the previous surface.
            for coll in self.ax.collections[:]:
                coll.remove()
            # Replot using the previously computed grid.
            self.surf = self.ax.plot_surface(self.X, self.Y, arr, cmap="viridis")
            self.ax.draw_artist(self.surf)
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()

        # Instead of repeatedly creating a new BytesIO object,
        # you might reuse one if thread-safety is ensured.
        buf = BytesIO()
        self.fig.savefig(buf, format="png")
        buf.seek(0)
        frame = buf.getbuffer()
        return frame


class OffscreenPyqtGraphSurfacePlotUpdater:
    def __init__(self):
        """
        Initialize the updater.
        A QApplication is created if necessary and the GLViewWidget is set up offscreen.
        The camera is also positioned far away ("zoomed out").
        """
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)

        # Create the GLViewWidget offscreen.
        self.view = gl.GLViewWidget()
        self.view.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
        self.view.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.view.resize(800, 600)

        # Zoom out. Adjust distance as needed.
        self.view.setCameraPosition(distance=4)

        # Add a grid for context.
        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        self.view.addItem(grid)

        # Reference to the surface plot item.
        self.surface = None

    def update(self, arr: NDArray[np.uint8]) -> memoryview:
        """
        Creates or updates the 3D surface plot from the 2D np.uint8 array.
        The height values (0–255) are normalized and scaled for a more pronounced relief.
        No external colormap is applied.

        The rendering is captured offscreen and returned as a memoryview of PNG image data.

        Parameters:
            arr (np.ndarray): A 2D numpy array of type np.uint8 used as the height map.

        Returns:
            memoryview: A memoryview of the PNG image data.
        """
        if arr.ndim != 2 or arr.dtype != np.uint8:
            raise ValueError("Input array must be 2-dimensional and of type np.uint8")

        nrows, ncols = arr.shape
        # Convert the input array to float32.
        z = arr.astype(np.float32)

        # Normalize the heights to [0, 1].
        norm = (z - z.min()) / (z.max() - z.min() + 1e-8)

        # Scale the normalized heights to exaggerate differences.
        scaled_z = norm * 10

        # Compute scales so that the x and y axes span [0, 1].
        xscale = 1.0 / (ncols - 1) if ncols > 1 else 1.0
        yscale = 1.0 / (nrows - 1) if nrows > 1 else 1.0

        # Create or update the GLSurfacePlotItem without additional color mapping.
        if self.surface is None:
            self.surface = gl.GLSurfacePlotItem(z=scaled_z, shader="shaded", smooth=False)
            self.surface.setGLOptions("opaque")
            self.surface.translate(0, 0, 0)
            self.surface.scale(xscale, yscale, 1)
            self.view.addItem(self.surface)
        else:
            self.surface.setData(z=scaled_z)
            self.surface.scale(xscale, yscale, 1)

        # Process events and allow a short delay for proper offscreen rendering.
        self.view.repaint()
        self.app.processEvents()
        time.sleep(0.1)

        if GL is not None:
            GL.glFinish()

        # Grab the current rendering from the offscreen widget.
        try:
            qimg = self.view.grabFramebuffer()
        except AttributeError:
            qimg = self.view.grab()

        # Save the QImage into a QBuffer in PNG format.
        qbuffer = QtCore.QBuffer()
        qbuffer.open(QtCore.QIODevice.ReadWrite)
        qimg.save(qbuffer, "PNG")
        qbuffer.seek(0)

        # Convert the QBuffer data to bytes and return as a memoryview.
        data = bytes(qbuffer.readAll())
        return memoryview(data)


class VisPySurfacePlotter:
    def __init__(self, data_shape):
        # Create a canvas without interactive keys
        self.canvas = scene.SceneCanvas(show=False)
        self.view = self.canvas.central_widget.add_view()

        # Create a surface plot
        self.surface = visuals.SurfacePlot(
            x=np.linspace(0, 1, data_shape[0]),
            y=np.linspace(0, 1, data_shape[1]),
            z=np.zeros(data_shape, dtype=np.float32),
            color=(0.5, 0.5, 1, 1),
            shading="smooth",
        )
        self.view.add(self.surface)

        # Set the camera
        self.view.camera = scene.TurntableCamera(up="z", azimuth=30, elevation=60, distance=5)

    def update(self, data: NDArray[np.uint8]):
        # Update the surface plot data
        self.surface.set_data(z=data.astype(np.float32))

        # Capture the buffer
        buffer = self.canvas.render(bgcolor="white", alpha=False)
        print(np.shape(buffer))
        contiguous_buffer = np.ascontiguousarray(buffer[..., :3]).astype(np.uint8)
        # Convert to memoryview
        buf = BytesIO(initial_bytes=contiguous_buffer.tobytes())

        # return contiguous_buffer
        return buf.getbuffer()


surface_plotter = VisPySurfacePlotter((1024, 1024))
# surface_plotter = OffscreenPyqtGraphSurfacePlotUpdater()


def surface_plot(image: NDArray[np.uint8]) -> memoryview:
    """
    Generates a 3D surface plot from a 2D np.uint8 array using matplotlib.
    The plot is rendered in a memory buffer for efficient access.

    Parameters:
        image (np.ndarray): A 2D numpy array of type np.uint8 used as the height values.

    Returns:
        io.BytesIO: A buffer containing the rendered surface plot in PNG format.
    """
    print(np.shape(image))
    return surface_plotter.update(image)


def heat_map(intensity_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    # timer = Timer(name="heat_map")
    flipped = np.flip(intensity_image, axis=0)
    img = np.empty((*flipped.shape, 3), dtype=np.uint8)
    img[..., 0] = np.where(flipped < 128, 0, (flipped - 128) * 2)  # red
    img[..., 1] = np.where(flipped < 128, flipped * 2, 255)  # green
    img[..., 2] = np.where(flipped < 128, 255 - flipped * 2, 0)  # blue
    # timer.log_time()
    return img


def create_frame(rgb_array: NDArray[np.uint8]):
    # timer: Timer = Timer(name="create_frame")
    buf: BytesIO = BytesIO()
    image: Image.Image = Image.fromarray(rgb_array, "RGB")
    image.save(buf, "JPEG")
    frame: memoryview[int] = buf.getbuffer()
    # timer.log_time()
    return frame


if __name__ == "__main__":
    # Define the sinc function
    def sinc(x, y):
        # Calculate the radius
        r = np.sqrt(x**2 + y**2)
        # Normalize the r to prevent division by zero
        # sinc(0) = 1
        z = np.sinc(r / np.pi)
        return z

    # Example usage
    data_shape = (1024, 1024)
    plotter = VisPySurfacePlotter(data_shape)
    x = np.linspace(-10, 10, data_shape[0])
    y = np.linspace(-10, 10, data_shape[1])
    x, y = np.meshgrid(x, y)
    z = sinc(x, y)
    print(np.shape(z))
    # data = np.random.randint(0, 256, data_shape, dtype=np.uint8)
    buf = plotter.update(z)
    image_data = np.frombuffer(buf, dtype=np.uint8).reshape(800, 600, 3)

    image = Image.fromarray(buf, "RGB")
    #
    image.show()
    # plotter.close()
