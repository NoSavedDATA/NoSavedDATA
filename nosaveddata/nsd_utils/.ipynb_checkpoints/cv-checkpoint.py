import os, cv2

def video_from_image_folder(image_folder, output_video_path, fps=30):
    #print(f"{image_folder}")
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Ensure images are in the correct order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'XVID' or 'MJPG' if 'mp4v' doesn't work

    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    #print(f"{video}")
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def frames_from_video(video_path, output_path, startframe=0):
    video = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = video.read()

        if not ret:
            break  # End of video
        if frame_count > startframe:

            # Process the frame (e.g., display, save)

            # Save the frame as an image
            cv2.imwrite(f"{output_path}/frame_{frame_count}.jpg", frame)

        frame_count += 1
    video.release()

def video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:

        ret, frame = video.read()

        if not ret:
            break  # End of video

        frame_count += 1
    
    video.release()
    print(f"{frame_count}")