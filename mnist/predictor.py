from tensorflow import keras
import numpy as np
from config import MODEL, num_classes

class Predictor:
    # Load the pre-trained model.
    model = keras.models.load_model(MODEL)
    print("Loaded model from disk")

    @staticmethod
    def predict(img, label):
        # Predictions vector
        predictions = Predictor.model.predict(img)

        predictions1 = list()
        confidences = list()
        for i in range(len(predictions)):
            preds = predictions[i]
            explabel = label[i]
            prediction1, prediction2 = np.argsort(-preds)[:2]

            # Activation level corresponding to the expected class
            confidence_expclass = preds[explabel]

            if prediction1 != explabel:
                confidence_notclass = preds[prediction1]
            else:
                confidence_notclass = preds[prediction2]

            confidence = confidence_expclass - confidence_notclass
            predictions1.append(prediction1)
            confidences.append(confidence)

        return predictions1, confidences

    @staticmethod
    def predict_single(img, label):
        explabel = label
        # Predictions vector
        predictions = Predictor.model.predict(img)

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        # Activation level corresponding to the expected class
        confidence_expclass = predictions[0][explabel]

        if prediction1 != label:
            confidence_notclass = predictions[0][prediction1]
        else:
            confidence_notclass = predictions[0][prediction2]

        confidence = confidence_expclass - confidence_notclass

        return prediction1, confidence

    @staticmethod
    def predict_generator(img, label):
        explabel = label
        # Predictions vector
        predictions = Predictor.model.predict(img, verbose=0)

        prediction1, prediction2 = np.argsort(-predictions[0])[:2]

        confidence_expclass = predictions[0][explabel]

        if prediction1 == label:
            success = True
            not_class = prediction2
            not_class_confidence = predictions[0][prediction2]
        else:
            success = False
            not_class = prediction1
            not_class_confidence = predictions[0][prediction1]

        return success, confidence_expclass, not_class, not_class_confidence

    @staticmethod
    def predict_datapoint(img, label):
        explabel = label
        # Predictions vector
        predictions = Predictor.model.predict(img, verbose=0)
        prediction1 = np.argsort(-predictions[0])[0]
        confidence_expclass = predictions[0][explabel]

        return (prediction1 == label), confidence_expclass, predictions[0]
