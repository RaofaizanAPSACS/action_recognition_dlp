import cv2
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import VGG16

# Load the pre-trained VGG16 model for feature extraction
vgg_model = VGG16(include_top=True, weights='imagenet')
vgg_model_transfer = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc2').output)

# Load the trained LSTM model
lstm_model = load_model('./models/lstm_model_v1.h5')

video_path = './videos/no255_xvid.avi'
output_path = './outputs/no255_xvid_out.avi'

def process_frame(image, img_size):
    # Resize and normalize the image
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Ensure the frame has batch dimension
    return image

def get_transfer_values(frame):
    # Predict the transfer values with VGG16
    transfer_values = vgg_model_transfer.predict(frame)
    return transfer_values

def predict_on_frames(transfer_values):
    # Prepare the shape for LSTM
    transfer_values = np.array(transfer_values)  # Shape should be (20, 4096)
    transfer_values = np.expand_dims(transfer_values, axis=0)  # Shape (1, 20, 4096)
    prediction = lstm_model.predict(transfer_values)
    return prediction

def main(video_path, output_path, img_size=224, frames_count=20):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        if out is None:
            # Initialize the video writer
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        processed_frame = process_frame(frame, img_size)

        if len(frame_buffer) < frames_count:
            frame_buffer.append(processed_frame)
        else:
            frame_buffer.pop(0)
            frame_buffer.append(processed_frame)

        if len(frame_buffer) == frames_count:
            transfer_values = [get_transfer_values(f) for f in frame_buffer]
            transfer_values = np.squeeze(np.array(transfer_values), axis=1)
            prediction = predict_on_frames(transfer_values)
            label = 'Violence' if prediction[0][0] > 0.5 else 'No Violence'

            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print("Video processing complete. Output saved to:", output_path)

if __name__ == "__main__":
    main(video_path, output_path)
