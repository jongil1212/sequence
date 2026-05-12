import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from io import BytesIO


st.set_page_config(
    page_title="시퀀스샷 생성기",
    page_icon="🎞️",
    layout="wide"
)


def get_video_info(video_path):
    """영상의 기본 정보를 가져옵니다."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height
    }


def resize_frame(frame, max_width=1000):
    """너무 큰 영상은 화면 표시와 처리 속도를 위해 크기를 줄입니다."""
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def extract_frames(video_path, interval_sec, start_sec, end_sec, max_width):
    """일정 시간 간격으로 프레임을 추출합니다."""
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    timestamps = []

    current_time = start_sec

    while current_time <= end_sec:
        frame_number = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if not ret:
            break

        frame = resize_frame(frame, max_width=max_width)

        # OpenCV는 BGR, PIL/Streamlit은 RGB를 사용하므로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame_rgb)
        timestamps.append(current_time)

        current_time += interval_sec

    cap.release()

    return frames, timestamps


def create_sequence_image(frames, alpha):
    """
    여러 프레임을 반투명하게 겹쳐 시퀀스샷 이미지를 만듭니다.
    첫 프레임을 기준 이미지로 두고, 이후 프레임들을 alpha 값으로 누적합니다.
    """
    if len(frames) == 0:
        return None

    base = frames[0].astype(np.float32)

    for frame in frames[1:]:
        overlay = frame.astype(np.float32)
        base = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

    result = np.clip(base, 0, 255).astype(np.uint8)

    return result


def create_contact_sheet(frames, timestamps, columns=4):
    """추출된 프레임들을 확인할 수 있는 미리보기 이미지를 만듭니다."""
    if len(frames) == 0:
        return None

    small_frames = []

    for frame, t in zip(frames, timestamps):
        img = Image.fromarray(frame)
        img.thumbnail((240, 180))

        canvas = Image.new("RGB", (240, 210), "white")
        x = (240 - img.width) // 2
        canvas.paste(img, (x, 0))

        small_frames.append((canvas, f"{t:.2f}초"))

    rows = int(np.ceil(len(small_frames) / columns))
    sheet = Image.new("RGB", (columns * 240, rows * 210), "white")

    for i, (img, label) in enumerate(small_frames):
        row = i // columns
        col = i % columns

        sheet.paste(img, (col * 240, row * 210))

        # 라벨은 간단히 생략해도 되지만, 여기서는 PIL 기본 기능만 사용
        # 한글 폰트 문제를 피하기 위해 시간 정보는 Streamlit 표에서 별도 제공

    return sheet


def image_to_bytes(image_array):
    """numpy 이미지를 PNG 다운로드용 bytes로 변환합니다."""
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("🎞️ 시퀀스샷 생성기")
st.write(
    "운동 영상을 업로드하면 일정한 시간 간격의 프레임을 추출하여 "
    "한 장의 이미지에 반투명하게 겹쳐 보여줍니다."
)

uploaded_file = st.file_uploader(
    "영상 파일을 업로드하세요.",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    video_info = get_video_info(video_path)

    if video_info is None:
        st.error("영상을 읽을 수 없습니다. 다른 파일 형식으로 다시 시도해 주세요.")
        st.stop()

    st.subheader("영상 정보")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("재생 시간", f"{video_info['duration']:.2f}초")
    with col2:
        st.metric("FPS", f"{video_info['fps']:.2f}")
    with col3:
        st.metric("전체 프레임", f"{video_info['frame_count']}")
    with col4:
        st.metric("해상도", f"{video_info['width']} × {video_info['height']}")

    st.video(uploaded_file)

    st.subheader("시퀀스샷 설정")

    duration = video_info["duration"]

    col1, col2 = st.columns(2)

    with col1:
        start_sec = st.number_input(
            "시작 시간(초)",
            min_value=0.0,
            max_value=float(duration),
            value=0.0,
            step=0.1
        )

    with col2:
        end_sec = st.number_input(
            "종료 시간(초)",
            min_value=0.0,
            max_value=float(duration),
            value=float(duration),
            step=0.1
        )

    interval_sec = st.slider(
        "프레임 추출 간격(초)",
        min_value=0.1,
        max_value=2.0,
        value=0.3,
        step=0.1
    )

    alpha = st.slider(
        "겹쳐지는 프레임의 투명도",
        min_value=0.05,
        max_value=0.80,
        value=0.30,
        step=0.05
    )

    max_width = st.selectbox(
        "출력 이미지 최대 너비",
        options=[600, 800, 1000, 1200, 1600],
        index=2
    )

    expected_frame_count = int((end_sec - start_sec) / interval_sec) + 1

    st.info(f"예상 추출 프레임 수: 약 {expected_frame_count}장")

    if end_sec <= start_sec:
        st.warning("종료 시간은 시작 시간보다 커야 합니다.")
        st.stop()

    if expected_frame_count > 80:
        st.warning(
            "추출 프레임 수가 너무 많습니다. "
            "처리 시간이 길어질 수 있으니 시간 간격을 늘리는 것을 권장합니다."
        )

    if st.button("시퀀스샷 생성하기", type="primary"):
        with st.spinner("프레임을 추출하고 시퀀스샷을 생성하는 중입니다..."):
            frames, timestamps = extract_frames(
                video_path=video_path,
                interval_sec=interval_sec,
                start_sec=start_sec,
                end_sec=end_sec,
                max_width=max_width
            )

            if len(frames) == 0:
                st.error("프레임을 추출하지 못했습니다. 설정을 다시 확인해 주세요.")
                st.stop()

            sequence_image = create_sequence_image(frames, alpha)

        st.success(f"{len(frames)}장의 프레임을 추출하여 시퀀스샷을 생성했습니다.")

        st.subheader("생성된 시퀀스샷")
        st.image(
            sequence_image,
            caption="반투명하게 겹친 시퀀스샷",
            use_container_width=True
        )

        png_bytes = image_to_bytes(sequence_image)

        st.download_button(
            label="시퀀스샷 PNG 다운로드",
            data=png_bytes,
            file_name="sequence_shot.png",
            mime="image/png"
        )

        st.subheader("추출된 프레임 확인")

        frame_table = {
            "프레임 번호": list(range(1, len(timestamps) + 1)),
            "시간(초)": [round(t, 2) for t in timestamps]
        }

        st.dataframe(frame_table, use_container_width=True)

        with st.expander("추출된 프레임 미리보기"):
            preview_cols = st.columns(4)

            for i, frame in enumerate(frames):
                with preview_cols[i % 4]:
                    st.image(
                        frame,
                        caption=f"{timestamps[i]:.2f}초",
                        use_container_width=True
                    )

else:
    st.info("왼쪽 또는 위의 업로드 영역에 운동 영상을 올려 주세요.")
