import os
import shutil
import sys

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model, load_image
from utils.preprocessor import preprocess_input

emojiMap = {
    "happy": [(0.9, "😁"), (0.8, "😄"), (0.7, "😃"), (0.5, "🙂")],
    "sad": [(0.9, "😭"), (0.8, "😢"), (0.7, "😔"), (0.5,"😞")],
    "fear": [(0.9, "😱"), (0.8, "😨"), (0.7, "😧"), (0.5," 😬")],
    "disgust": [(0.9, " 🤮"), (0.8, "🤢"), (0.7, "😖"), (0.5,"😒")],
    "angry": [(0.9, "😡"), (0.8, "😤"), (0.5, "😠")],
    "surprise": [(0.9, "😲"), (0.8, "😯"), (0.7, "😮"), (0.5,"🫢")],
    "neutral": [(0.1, "😐")]
}

THRESHOLD = 0.5

def map_emotions_to_emojis(emotions, threshold=0.5):
    """
    Maps emotions with probabilities above a threshold to corresponding emojis based on intensity levels.
    """
    plausible_emojis = []
    for emotion, probability in emotions.items():
        if probability > threshold:
            # Look up the intensity levels and emojis for the current emotion
            intensity_levels = emojiMap.get(emotion, [])
            # Default to the least intense emoji if none match
            selected_emoji = intensity_levels[-1][1] if intensity_levels else ''
            # Iterate through the intensity levels for the emotion
            for intensity_threshold, emoji in intensity_levels:
                if probability >= intensity_threshold:
                    selected_emoji = emoji
                    break  # Found the matching intensity level, no need to continue
            if selected_emoji:  # Add the selected emoji if it's not empty
                plausible_emojis.append(selected_emoji)
    return plausible_emojis

def emotion_count(list_):
    """Counts the occurrence of each emotion in the list."""
    count = {}
    for emotion in list_:
        count[emotion] = count.get(emotion, 0) + 1
    return count

def get_emotion_distribution(dict_):
    """Generates a distribution of emotions from the input dictionary."""
    emotions = []
    for frame_nmr in dict_.keys():
        for face_nmr in dict_[frame_nmr].keys():
            emotions.append(dict_[frame_nmr][face_nmr]['predicted_emotion'])

    return emotion_count(emotions)

def process():
    # parameters for loading data and images
    image_path = sys.argv[1]
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    emotion_offsets = (0, 0)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    frames_dir = './.tmp'
    if image_path[-3:] in ['jpg', 'png']:
        images_list = [image_path]
    else:
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.mkdir(frames_dir)
        os.system('ffmpeg -i {} {}/$frame_%010d.jpg'.format(image_path, frames_dir))
        images_list = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))]

    output = {}
    # if there are multiple input images
    for image_path_index, image_path in enumerate(images_list):
        # loading images
        gray_image = load_image(image_path, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        faces = detect_faces(face_detection, gray_image)

        tmp = {}
        for face_coordinates_index, face_coordinates in enumerate(faces):
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_predictions = emotion_classifier.predict(gray_face)[0]
            emotion_label_arg = np.argmax(emotion_predictions)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_score = np.amax(emotion_predictions)

            # Constructing the emotion distribution
            emotion_distribution = {emotion_labels[i]: float(emotion_predictions[i]) for i in range(len(emotion_labels))}

            plausible_emojis = map_emotions_to_emojis(emotion_distribution, THRESHOLD)

            tmp[face_coordinates_index] = {
                'predicted_emotion': emotion_text, 
                'score': emotion_score,
                'location': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},  # Face location
                'emotions': emotion_distribution,  # Emotion distribution
                'emojis': plausible_emojis  # Added: Emojis for plausible emotions

            }

        output[image_path_index] = tmp

    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)

    return output, get_emotion_distribution(output)


if __name__ == "__main__":
    output, emotion_distribution = process()

    for key in output.keys():
        print(output[key])

    print(emotion_distribution)
