import numpy as np
from keras.models import load_model
import cv2


def predict(x):
    # Here x is a NumPy array. On the actual exam it will be a list of paths.
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    imgs = []
    for file in x:
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(folder, file.name))
            img_resize = cv2.resize((100, 100), Image.NEAREST)
            # img_resize = img.resize((64, 64), Image.BILINEAR)
            # img_resize = img.resize((64, 64), Image.BICUBIC)
            # img_resize = img.resize((64, 64), Image.ANTIALIAS)
            img_resize = np.array(img_resize)
            imgs.append(img_resize.flatten())
    x = np.array(imgs).reshape(len(imgs), 30000)
    # Write any data prep you used during training
    # %% --------------------------------------------- Predict ---------------------------------------------------------
    assert isinstance(y_test_pred, type(np.array([1])))  # Checks if your returned y_test_pred is a NumPy array
    assert y_test_pred.shape == (len(x_test),)  # Checks if its shape is this one (one label per image path)
    # Checks whether the range of your predicted labels is correct
    assert np.unique(y_test_pred).max() <= 3 and np.unique(y_test_pred).min() >= 0

    model = load_model('mlp_tan1300549644.hdf5')
    y_pred = np.argmax(model.predict(x), axis=1)
    return y_pred, model
