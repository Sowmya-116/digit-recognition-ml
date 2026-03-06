import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

model = load_model("digit_model.h5")

(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

index = 10

sample = x_test[index]
sample = sample.reshape(1, 28, 28, 1)

prediction = model.predict(sample)
predicted_digit = np.argmax(prediction)

print("Predicted digit:", predicted_digit)
print("Actual digit:", y_test[index])