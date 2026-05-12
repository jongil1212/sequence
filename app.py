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
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def read_frame_at_time(video_path, time_sec, max_width):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame = resize_frame(frame, max_width=max_width)
    return frame


def extract_frames(video_path, interval_sec, start_sec, end_sec, max_width):
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
        frames.append(frame)
        timestamps.append(current_time)

        current_time += interval_sec

    cap.release()

    return frames, timestamps


def create_motion_mask(background, frame, threshold, blur_size, dilate_iter, min_area):
    """
    배경 프레임과 현재 프레임의 차이를 이용해 움직이는 부분만 마스크로 만듭니다.
    흰색 부분이 움직이는 영역입니다.
    """
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(bg_gray, frame_gray)

    if blur_size > 1:
        if blur_size % 2 == 0:
            blur_size += 1
        diff = cv2.GaussianBlur(diff, (blur_size, blur_size), 0)

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 너무 작은 노이즈 제거
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return clean_mask


def overlay_motion_parts(background, frames, threshold, blur_size, dilate_iter, min_area, opacity):
    """
    배경은 고정하고, 각 프레임에서 움직이는 부분만 추출해 누적 합성합니다.
    """
    result = background.copy().astype(np.float32)

    for frame in frames:
        mask = create_motion_mask(
            background=background,
            frame=frame,
            threshold=threshold,
            blur_size=blur_size,
            dilate_iter=dilate_iter,
            min_area=min_area
        )

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bool = mask_3ch > 0

        frame_float = frame.astype(np.float32)

        # 움직이는 부분만 result 위에 opacity만큼 합성
        result[mask_bool] = (
            result[mask_bool] * (1 - opacity)
            + frame_float[mask_bool] * opacity
        )

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def create_simple_blend(frames, alpha):
    """
    비교용: 기존 방식처럼 전체 프레임을 반투명하게 겹칩니다.
    """
    if len(frames) == 0:
        return None

    base = frames[0].astype(np.float32)

    for frame in frames[1:]:
        overlay = frame.astype(np.float32)
        base = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

    return np.clip(base, 0, 255).astype(np.uint8)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_bytes_bgr(image_bgr):
    image_rgb = bgr_to_rgb(image_bgr)
    img = Image.fromarray(image_rgb)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


st.title("🎞️ 시퀀스샷 생성기")
st.write(
    "운동 영상을 업로드하면 일정한 시간 간격의 프레임을 추출하고, "
    "움직이는 부분만 배경 위에 겹쳐 시퀀스샷을 만듭니다."
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

    col1, col2, col3 = st.columns(3)

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

    with col3:
        background_sec = st.number_input(
            "배경으로 사용할 시간(초)",
            min_value=0.0,
            max_value=float(duration),
            value=0.0,
            step=0.1,
            help="움직이는 물체가 가장 적게 보이는 장면을 배경으로 선택하면 좋습니다."
        )

    interval_sec = st.slider(
        "프레임 추출 간격(초)",
        min_value=0.03,
        max_value=1.0,
        value=0.12,
        step=0.01
    )

    max_width = st.selectbox(
        "출력 이미지 최대 너비",
        options=[600, 800, 1000, 1200, 1600],
        index=2
    )

    st.subheader("움직이는 부분 추출 설정")

    col1, col2 = st.columns(2)

    with col1:
        threshold = st.slider(
            "움직임 감지 민감도",
            min_value=5,
            max_value=80,
            value=25,
            step=1,
            help="값이 낮을수록 작은 움직임까지 감지합니다. 배경 노이즈가 많으면 값을 높이세요."
        )

        blur_size = st.slider(
            "노이즈 완화 정도",
            min_value=1,
            max_value=21,
            value=5,
            step=2,
            help="값이 클수록 자잘한 노이즈가 줄지만, 물체 경계가 흐려질 수 있습니다."
        )

    with col2:
        dilate_iter = st.slider(
            "움직이는 영역 확장",
            min_value=0,
            max_value=8,
            value=2,
            step=1,
            help="값이 클수록 감지된 물체 영역이 조금 더 넓어집니다."
        )

        min_area = st.slider(
            "작은 점 노이즈 제거",
            min_value=0,
            max_value=5000,
            value=300,
            step=50,
            help="이 값보다 작은 움직임 영역은 제거합니다."
        )

    opacity = st.slider(
        "움직이는 물체의 진하기",
        min_value=0.1,
        max_value=1.0,
        value=0.85,
        step=0.05
    )

    show_comparison = st.checkbox(
        "기존 전체 반투명 합성 결과도 함께 보기",
        value=False
    )

    expected_frame_count = int((end_sec - start_sec) / interval_sec) + 1

    st.info(f"예상 추출 프레임 수: 약 {expected_frame_count}장")

    if end_sec <= start_sec:
        st.warning("종료 시간은 시작 시간보다 커야 합니다.")
        st.stop()

    if expected_frame_count > 100:
        st.warning(
            "추출 프레임 수가 너무 많습니다. "
            "처리 시간이 길어질 수 있으니 시간 간격을 늘리는 것을 권장합니다."
        )

    if st.button("시퀀스샷 생성하기", type="primary"):
        with st.spinner("프레임을 추출하고 움직이는 부분을 합성하는 중입니다..."):
            background = read_frame_at_time(
                video_path=video_path,
                time_sec=background_sec,
                max_width=max_width
            )

            if background is None:
                st.error("배경 프레임을 읽지 못했습니다.")
                st.stop()

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

            sequence_image = overlay_motion_parts(
                background=background,
                frames=frames,
                threshold=threshold,
                blur_size=blur_size,
                dilate_iter=dilate_iter,
                min_area=min_area,
                opacity=opacity
            )

            if show_comparison:
                simple_blend = create_simple_blend(frames, alpha=0.3)

        st.success(f"{len(frames)}장의 프레임을 사용하여 시퀀스샷을 생성했습니다.")

        st.subheader("생성된 시퀀스샷")

        st.image(
            bgr_to_rgb(sequence_image),
            caption="배경 고정 + 움직이는 부분만 합성한 시퀀스샷",
            use_container_width=True
        )

        png_bytes = image_to_bytes_bgr(sequence_image)

        st.download_button(
            label="시퀀스샷 PNG 다운로드",
            data=png_bytes,
            file_name="sequence_shot_motion_only.png",
            mime="image/png"
        )

        if show_comparison:
            st.subheader("비교: 기존 전체 반투명 합성 방식")

            st.image(
                bgr_to_rgb(simple_blend),
                caption="전체 프레임을 반투명하게 겹친 결과",
                use_container_width=True
            )

        st.subheader("배경 프레임 확인")

        st.image(
            bgr_to_rgb(background),
            caption=f"배경 프레임: {background_sec:.2f}초",
            use_container_width=True
        )

        st.subheader("추출된 프레임 시간")

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
                        bgr_to_rgb(frame),
                        caption=f"{timestamps[i]:.2f}초",
                        use_container_width=True
                    )

        with st.expander("설정 조정 팁"):
            st.write(
                """
                결과가 잘 안 나오면 아래처럼 조정해 보세요.

                - 배경이 지저분하게 같이 잡히면: **움직임 감지 민감도 값을 올리기**
                - 물체 일부가 사라지면: **움직임 감지 민감도 값을 낮추기**
                - 물체가 너무 얇게 잡히면: **움직이는 영역 확장 값을 올리기**
                - 작은 점들이 많이 보이면: **작은 점 노이즈 제거 값을 올리기**
                - 물체가 너무 흐리면: **움직이는 물체의 진하기 값을 올리기**
                - 궤적이 너무 촘촘하면: **프레임 추출 간격을 늘리기**
                - 궤적이 너무 듬성하면: **프레임 추출 간격을 줄이기**
                """
            )

else:
    st.info("운동 영상을 업로드해 주세요.")
