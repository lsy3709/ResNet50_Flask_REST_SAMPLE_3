import cv2
import os
from services.model_loader import yolo_model


def process_yolo(file_path, output_path, file_type):
    """
    YOLO 모델을 사용하여 이미지 또는 동영상을 처리합니다.
    이 함수는 별도의 스레드에서 실행됩니다.
    """
    try:
        if file_type == 'image':
            results = yolo_model(file_path)
            result_img = results[0].plot()
            cv2.imwrite(output_path, result_img)
            print(f"✅ [YOLO Image] 처리 완료: {output_path}")

        elif file_type == 'video':
            process_video(file_path, output_path)

    except Exception as e:
        print(f"🔴 ERROR in process_yolo thread: {e}")


def process_video(file_path, output_path):
    """비디오 파일을 프레임별로 처리하고 결과를 저장합니다."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"🔴 ERROR: 비디오 파일을 열 수 없습니다: {file_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30  # FPS 감지 실패 시 기본값 설정

    # 비디오 저장을 위한 VideoWriter 설정 (mp4v 코덱 사용)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"🔴 ERROR: VideoWriter를 초기화할 수 없습니다. 코덱 또는 경로를 확인하세요.")
        cap.release()
        return

    print(f"🚀 [YOLO Video] 처리 시작: {file_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model(frame)
        result_frame = results[0].plot()
        out.write(result_frame)

    cap.release()
    out.release()
    print(f"✅ [YOLO Video] 처리 완료: {output_path}")
