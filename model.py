import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array

img_height, img_width = 128, 128
class_labels = ["biodegradable", "recyclable", "hazardous", "general"]

def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(class_labels), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

X_dummy = np.random.rand(20, img_height, img_width, 3)  
y_dummy = np.random.randint(0, 4, 20)                   
y_dummy = np.eye(4)[y_dummy]                            

model = create_model()
model.fit(X_dummy, y_dummy, epochs=2)  


model.save("waste_classifier.h5")
print("Model saved as 'waste_classifier.h5'")

disposal_guidance = {
    "biodegradable": {
        "instructions": [
            "Place in the GREEN bin (organic waste).",
            "Compost food and garden waste at home.",
            "Avoid mixing with plastics or metals."
        ],
        "eco_impact": "Composting biodegradable waste reduces methane emissions and creates natural fertilizer."
    },
    "recyclable": {
        "instructions": [
            "Place in the BLUE bin (dry recyclables).",
            "Clean and dry items before disposal.",
            "Take large recyclables to the nearest recycling center."
        ],
        "eco_impact": "Recycling 1 ton of plastic saves up to 1,500 liters of oil and reduces CO2 emissions."
    },
    "hazardous": {
        "instructions": [
            "Do NOT throw in regular bins.",
            "Take to a hazardous waste disposal facility or e-waste collection point.",
            "Handle with gloves and store safely until disposal."
        ],
        "eco_impact": "Proper hazardous waste disposal prevents soil and water contamination."
    },
    "general": {
        "instructions": [
            "Place in the BLACK bin (non-recyclable waste).",
            "Avoid mixing with recyclable or compostable items.",
            "Reduce generation by reusing items where possible."
        ],
        "eco_impact": "Minimizing general waste helps reduce landfill load and methane gas emissions."
    }
}

def predict_and_guide(img_path, class_labels):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = class_labels[class_index]

    print("\nWaste Classification Result")
    print(f"Predicted Waste Type: {predicted_class.upper()}\n")

    if predicted_class in disposal_guidance:
        print("Disposal Instructions:")
        for step in disposal_guidance[predicted_class]["instructions"]:
            print(f"- {step}")
        
        print("\nEnvironmental Impact:")
        print(disposal_guidance[predicted_class]["eco_impact"])
    else:
        print("No guidance available for this waste type.")

    return predicted_class

test_image_path = "test.jpeg"
predicted_class = predict_and_guide(test_image_path, class_labels)
