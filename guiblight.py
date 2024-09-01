import keras
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy

from keras.models import load_model

model = load_model('models/MyModelTF.keras')
#dictionary to label all traffic signs class.
# ['EarlyBlight', 'Healthy', 'LateBlight']
classes = { 
    0: 'Early Blight',
    1: 'Healthy Leaf',
    2: 'Late Blight'
}

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Tomato Blight Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((228, 228))
    image = keras.utils.img_to_array(image)
    image = keras.ops.expand_dims(image, axis=0)
    # image = numpy.array(image)
    # image = image/255
    # pred = model.predict([image]).argmax(axis=-1)  # [0]
    pred = model.predict([image])
    print(pred)
    score = float(keras.ops.sigmoid(pred[0]))
    # score = float(keras.ops.sigmoid(pred))
    fpath = file_path.split("/")[-2]
    sign = f"This image is {100 * (1 - score):.2f}% Blighted and {100 * score:.2f}% Healthy."
    if fpath in ['Late Blight', 'Early Blight', 'Healthy']:
        sign = f"This image is {100 * score:.2f}% " + fpath
        if fpath == "Healthy":
            pass
    print(pred[0], score, file_path)
    # sign = classes[pred]
    # print(sign)
    label.configure(foreground='#011638', text=sign)

def show_classify_button(file_path):
    classify_b=Button(
        top,
        text="Classify Image",
        command=lambda: classify(file_path),
        padx=10,pady=5)

    classify_b.configure(
        background='#364156',
        foreground='white',
        font=('arial',10,'bold'))

    classify_b.place(
        relx=0.79,
        rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Upload an image", command=upload_image, padx=10,pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10,'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Tomato Blight Classification", pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
