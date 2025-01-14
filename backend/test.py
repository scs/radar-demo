import io
import tkinter as tk

from PIL import Image, ImageTk

from app.logic.radar_simulation import gen_frames
from app.logic.state import GlobalState


class ImageStreamApp:
    def __init__(self, master: tk.Tk) -> None:
        self.master: tk.Tk = master
        self.master.title("Image Streams")
        self.page_left = True

        # Create a frame that will contain the canvas
        self.frame: tk.Frame = tk.Frame(master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create a 2x2 grid of canvases for displaying images
        self.canvases = [[tk.Canvas(self.frame) for _ in range(2)] for _ in range(2)]
        for row in range(2):
            for col in range(2):
                self.canvases[row][col].grid(row=row, column=col, sticky="nsew")

        # Configure grid weights for resizing
        for i in range(2):
            self.frame.grid_rowconfigure(i, weight=1)
            self.frame.grid_columnconfigure(i, weight=1)
        # Create image generators
        self.generators = [gen_frames(i) for i in range(4)]

        # Store PhotoImage references
        self.photo_images = [None] * 4

        # Add a button to start the image streams
        self.start_button = tk.Button(master, text="Start", command=self.start)
        self.start_button.pack()
        # Add a button to start the image streams
        self.stop_button = tk.Button(master, text="Stop", command=self.stop)
        self.stop_button.pack()

        # Start updating images
        self.update_images()

    def start(self) -> None:
        GlobalState.set_running()

    def stop(self) -> None:
        print("Setting to stopping")
        GlobalState.set_stopped()

    def update_images(self) -> None:
        for i in range(2):
            for j in range(2):
                canvas = self.canvases[i][j]
                linear_index = 2 * i + j
                # Get next image from the generator
                data = next(self.generators[linear_index])

                # Find the start of the JPEG image data
                header_end = data.find(b"\r\n\r\n") + 4
                image_data = data[header_end:-2]  # Exclude the trailing "\r\n"

                # Load the image from the byte data
                image: Image.Image = Image.open(io.BytesIO(image_data))

                # Resize image to fit the canvas while maintaining aspect ratio
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                size = min(canvas_width, canvas_height)
                image = image.resize((size, size))

                # Display the image
                self.photo_images[linear_index] = ImageTk.PhotoImage(image)
                _ = canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_images[linear_index])

        # Schedule the next update
        _ = GlobalState.gen_frame_number()
        _ = self.master.after(1, self.update_images)


if __name__ == "__main__":
    root = tk.Tk()

    GlobalState.init_state("QUAD_CORNER")
    app = ImageStreamApp(root)

    # Make the window resizable
    root.geometry("800x800")
    root.mainloop()
