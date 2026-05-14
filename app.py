from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st


VIOLET_HUE = 145
VIDEO_EXTENSIONS = {"mp4"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png"}


@dataclass(frozen=True)
class ProcessingSettings:
    dilation_iterations: int
    min_saturation: int
    min_value: int
    red_dominance: int


@dataclass(frozen=True)
class ExportPreset:
    label: str
    suffix: str
    image_size: Optional[Tuple[int, int]]
    video_size: Optional[Tuple[int, int]]


EXPORT_PRESETS = {
    "original": ExportPreset("Tel quel", "original", None, None),
    "linkedin": ExportPreset("LinkedIn feed", "linkedin", (1200, 628), (1920, 1080)),
    "instagram": ExportPreset("Instagram feed", "instagram", (1080, 1350), (1080, 1350)),
    "facebook": ExportPreset("Facebook feed", "facebook", (1080, 1350), (1080, 1350)),
}
EXPORT_BUTTONS = (
    ("original", "+", "Aucun changement\nde format"),
    ("facebook", "f", "Facebook"),
    ("instagram", "◎", "Instagram"),
    ("linkedin", "in", "LinkedIn"),
)


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def uploaded_extension(uploaded_file) -> str:
    return os.path.splitext(uploaded_file.name)[1].lower().lstrip(".")


def is_video_file(uploaded_file) -> bool:
    return uploaded_extension(uploaded_file) in VIDEO_EXTENSIONS


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    image_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    decoded = cv2.imdecode(image_bytes, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Impossible de lire cette image.")

    if decoded.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)

    if decoded.shape[2] == 4:
        rgb = cv2.cvtColor(decoded[:, :, :3], cv2.COLOR_BGR2RGB)
        return np.dstack((rgb, decoded[:, :, 3]))

    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def process_image(image: np.ndarray, settings: ProcessingSettings) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        processed_rgb = recolor_blood_frame(image[:, :, :3], settings)
        return np.dstack((processed_rgb, image[:, :, 3]))
    return recolor_blood_frame(image, settings)


def resize_cover(rgb_image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    scale = max(target_width / width, target_height / height)
    resized = cv2.resize(rgb_image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    start_x = max((resized.shape[1] - target_width) // 2, 0)
    start_y = max((resized.shape[0] - target_height) // 2, 0)
    return resized[start_y : start_y + target_height, start_x : start_x + target_width]


def resize_contain(rgb_image: np.ndarray, target_width: int, target_height: int) -> Tuple[np.ndarray, int, int]:
    height, width = rgb_image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = round(width * scale)
    new_height = round(height * scale)
    resized = cv2.resize(rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    return resized, x_offset, y_offset


def adapt_to_social_canvas(image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    if target_size is None:
        return image

    target_width, target_height = target_size
    has_alpha = image.ndim == 3 and image.shape[2] == 4
    rgb_image = image[:, :, :3] if has_alpha else image

    background = resize_cover(rgb_image, target_width, target_height)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=18, sigmaY=18)
    foreground, x_offset, y_offset = resize_contain(rgb_image, target_width, target_height)

    canvas = background.copy()
    if has_alpha:
        foreground_alpha = cv2.resize(
            image[:, :, 3],
            (foreground.shape[1], foreground.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        alpha = (foreground_alpha.astype(np.float32) / 255.0)[:, :, None]
        region = canvas[y_offset : y_offset + foreground.shape[0], x_offset : x_offset + foreground.shape[1]]
        canvas[y_offset : y_offset + foreground.shape[0], x_offset : x_offset + foreground.shape[1]] = (
            region.astype(np.float32) * (1.0 - alpha) + foreground.astype(np.float32) * alpha
        ).astype(np.uint8)
    else:
        canvas[y_offset : y_offset + foreground.shape[0], x_offset : x_offset + foreground.shape[1]] = foreground

    return canvas


def encode_image(image: np.ndarray, source_name: str, preset: ExportPreset) -> Tuple[bytes, str, str]:
    source_extension = os.path.splitext(source_name)[1].lower().lstrip(".")
    has_alpha = image.ndim == 3 and image.shape[2] == 4
    use_png = preset.image_size is None and (has_alpha or source_extension == "png")
    stem = os.path.splitext(source_name)[0]

    if use_png:
        output_name = f"humcolor_{preset.suffix}_{stem}.png"
        mime = "image/png"
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA) if has_alpha else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".png", encoded_image)
    else:
        output_name = f"humcolor_{preset.suffix}_{stem}.jpg"
        mime = "image/jpeg"
        encoded_image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", encoded_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    if not ok:
        raise ValueError("Impossible d'encoder l'image transformee.")
    return buffer.tobytes(), output_name, mime


def first_video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        raise ValueError("Aucun flux vidéo n'a été trouvé dans ce fichier.")
    return stream


def stream_framerate(stream: av.video.stream.VideoStream) -> Optional[Fraction]:
    return (
        stream.average_rate
        or getattr(stream, "base_rate", None)
        or getattr(stream, "guessed_rate", None)
        or Fraction(30, 1)
    )


def stream_bitrate(stream: av.video.stream.VideoStream, container_bitrate: Optional[int]) -> Optional[int]:
    return stream.bit_rate or container_bitrate


def make_red_mask(rgb_frame: np.ndarray, hsv: np.ndarray, settings: ProcessingSettings) -> np.ndarray:
    lower_red_a = np.array([0, settings.min_saturation, settings.min_value], dtype=np.uint8)
    upper_red_a = np.array([10, 255, 255], dtype=np.uint8)
    lower_red_b = np.array([160, settings.min_saturation, settings.min_value], dtype=np.uint8)
    upper_red_b = np.array([180, 255, 255], dtype=np.uint8)

    mask_a = cv2.inRange(hsv, lower_red_a, upper_red_a)
    mask_b = cv2.inRange(hsv, lower_red_b, upper_red_b)
    mask = cv2.bitwise_or(mask_a, mask_b)

    red = rgb_frame[:, :, 0].astype(np.int16)
    green = rgb_frame[:, :, 1].astype(np.int16)
    blue = rgb_frame[:, :, 2].astype(np.int16)
    red_dominance_mask = ((red - green) >= settings.red_dominance) & ((red - blue) >= settings.red_dominance)
    mask = cv2.bitwise_and(mask, (red_dominance_mask.astype(np.uint8) * 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if settings.dilation_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=settings.dilation_iterations)

    return cv2.GaussianBlur(mask, (7, 7), 0)


def recolor_blood_frame(rgb_frame: np.ndarray, settings: ProcessingSettings) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    mask = make_red_mask(rgb_frame, hsv, settings)

    violet_hsv = hsv.copy()
    violet_hsv[:, :, 0] = VIOLET_HUE
    recolored_rgb = cv2.cvtColor(violet_hsv, cv2.COLOR_HSV2RGB)

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = rgb_frame.astype(np.float32) * (1.0 - alpha) + recolored_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def frame_generator(input_path: str, settings: ProcessingSettings) -> Iterator[Tuple[av.VideoFrame, np.ndarray]]:
    with av.open(input_path) as input_container:
        video_stream = first_video_stream(input_container)
        for frame in input_container.decode(video_stream):
            rgb = frame.to_ndarray(format="rgb24")
            processed = recolor_blood_frame(rgb, settings)
            yield frame, processed


def extract_preview(input_path: str, settings: ProcessingSettings) -> Tuple[np.ndarray, np.ndarray]:
    with av.open(input_path) as input_container:
        video_stream = first_video_stream(input_container)
        frame = next(input_container.decode(video_stream), None)
        if frame is None:
            raise ValueError("Impossible de lire une image depuis cette vidéo.")

        before = frame.to_ndarray(format="rgb24")
        after = recolor_blood_frame(before, settings)
        return before, after


def codec_name_for_output(source_stream: av.video.stream.VideoStream) -> str:
    codec_name = source_stream.codec_context.name
    if codec_name in {"h264", "hevc", "mpeg4", "vp9", "av1"}:
        return codec_name
    return "h264"


def configure_output_stream(
    output_stream: av.video.stream.VideoStream,
    source_stream: av.video.stream.VideoStream,
    source_container: av.container.InputContainer,
    target_size: Optional[Tuple[int, int]],
) -> None:
    if target_size:
        output_stream.width = target_size[0]
        output_stream.height = target_size[1]
    else:
        output_stream.width = source_stream.codec_context.width or source_stream.width
        output_stream.height = source_stream.codec_context.height or source_stream.height
    output_stream.pix_fmt = source_stream.codec_context.pix_fmt or "yuv420p"

    bitrate = stream_bitrate(source_stream, source_container.bit_rate)
    if bitrate:
        output_stream.bit_rate = bitrate

    if source_stream.codec_context.profile:
        try:
            output_stream.codec_context.profile = source_stream.codec_context.profile
        except (AttributeError, ValueError, TypeError):
            output_stream.codec_context.options = {
                **dict(output_stream.codec_context.options or {}),
                "profile": str(source_stream.codec_context.profile),
            }

    if source_stream.time_base:
        output_stream.time_base = source_stream.time_base

    if source_stream.codec_context.options:
        output_stream.codec_context.options = {
            **dict(source_stream.codec_context.options),
            **dict(output_stream.codec_context.options or {}),
        }


def process_video(
    input_path: str,
    output_path: str,
    settings: ProcessingSettings,
    preset: ExportPreset,
    progress_bar,
) -> None:
    with av.open(input_path) as input_container:
        source_stream = first_video_stream(input_container)
        total_frames = source_stream.frames or 0
        framerate = stream_framerate(source_stream)
        codec_name = codec_name_for_output(source_stream)

        with av.open(output_path, mode="w") as output_container:
            try:
                output_stream = output_container.add_stream(codec_name, rate=framerate)
            except Exception:
                if codec_name == "h264":
                    raise
                output_stream = output_container.add_stream("h264", rate=framerate)
            configure_output_stream(output_stream, source_stream, input_container, preset.video_size)

            processed_count = 0
            for source_frame, processed_rgb in frame_generator(input_path, settings):
                processed_rgb = adapt_to_social_canvas(processed_rgb, preset.video_size)
                output_frame = av.VideoFrame.from_ndarray(processed_rgb, format="rgb24")
                output_frame = output_frame.reformat(
                    width=output_stream.width,
                    height=output_stream.height,
                    format=output_stream.pix_fmt,
                )
                output_frame.pts = source_frame.pts
                output_frame.time_base = source_frame.time_base

                for packet in output_stream.encode(output_frame):
                    output_container.mux(packet)

                processed_count += 1
                if total_frames:
                    progress_bar.progress(min(processed_count / total_frames, 1.0))
                elif processed_count % 10 == 0:
                    progress_bar.progress(0.0, text=f"{processed_count} images traitees...")

            for packet in output_stream.encode():
                output_container.mux(packet)

    progress_bar.progress(1.0)


def remove_temp_file(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def render_export_buttons() -> ExportPreset:
    if "export_key" not in st.session_state:
        st.session_state.export_key = "original"

    st.caption("Format d'export")
    columns = st.columns(len(EXPORT_BUTTONS))
    for column, (key, icon, label) in zip(columns, EXPORT_BUTTONS):
        selected = st.session_state.export_key == key
        button_label = f"{icon}\n\n{label}"
        if column.button(
            button_label,
            key=f"export_{key}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            st.session_state.export_key = key
            st.rerun()

    return EXPORT_PRESETS[st.session_state.export_key]


def main() -> None:
    st.set_page_config(page_title="Hum.Color", layout="wide")
    st.markdown(
        """
        <style>
        div[class*="st-key-export_"] button {
            min-height: 88px;
            white-space: pre-line;
            font-weight: 700;
            line-height: 1.15;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Hum.Color")

    uploaded_file = st.file_uploader(
        "Selectionnez une video MP4 ou une photo chirurgicale",
        type=sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS),
    )
    dilation = st.slider(
        "Dilatation du masque",
        min_value=0,
        max_value=8,
        value=1,
        help="Augmente la couverture du violet autour des reflets et des zones rouges limites.",
    )
    min_saturation = st.slider(
        "Saturation minimale du sang",
        min_value=0,
        max_value=255,
        value=95,
        help="Augmentez cette valeur si la peau, les gants ou les ombres chaudes deviennent violets.",
    )
    red_dominance = st.slider(
        "Dominance rouge minimale",
        min_value=0,
        max_value=120,
        value=35,
        help="Exige que le canal rouge soit plus fort que les canaux vert et bleu.",
    )
    min_value = st.slider(
        "Luminosite minimale",
        min_value=0,
        max_value=255,
        value=45,
        help="Evite de recolorer certains bruits sombres, tout en gardant les zones de sang peu eclairees.",
    )
    if "export_key" not in st.session_state:
        st.session_state.export_key = "original"
    export_preset = EXPORT_PRESETS[st.session_state.export_key]

    if uploaded_file is not None:
        settings = ProcessingSettings(
            dilation_iterations=dilation,
            min_saturation=min_saturation,
            min_value=min_value,
            red_dominance=red_dominance,
        )

        if is_video_file(uploaded_file):
            input_path = save_uploaded_file(uploaded_file)
            output_path = None

            try:
                before_preview, after_preview = extract_preview(input_path, settings)
                after_preview = adapt_to_social_canvas(after_preview, export_preset.video_size)
                st.subheader("Avant / Apres")
                before_col, after_col = st.columns(2)
                before_col.image(before_preview, caption="Avant", use_container_width=True)
                after_col.image(after_preview, caption="Apres", use_container_width=True)
                export_preset = render_export_buttons()

                if st.button("Transformer la video", type="primary"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_tmp:
                        output_path = output_tmp.name

                    progress_bar = st.progress(0.0)
                    with st.spinner("Traitement frame par frame en cours..."):
                        process_video(input_path, output_path, settings, export_preset, progress_bar)

                    with open(output_path, "rb") as processed_file:
                        processed_bytes = processed_file.read()

                    remove_temp_file(output_path)
                    output_path = None

                    st.download_button(
                        "Telecharger la video transformee",
                        data=processed_bytes,
                        file_name=f"humcolor_{export_preset.suffix}_{uploaded_file.name}",
                        mime="video/mp4",
                    )

                    st.video(processed_bytes)

            except Exception as exc:
                st.error(f"Le traitement a echoue : {exc}")
            finally:
                remove_temp_file(input_path)
                remove_temp_file(output_path)

        else:
            try:
                before_preview = decode_uploaded_image(uploaded_file)
                processed_image = process_image(before_preview, settings)
                after_preview = adapt_to_social_canvas(processed_image, export_preset.image_size)
                image_bytes, output_name, mime = encode_image(after_preview, uploaded_file.name, export_preset)

                st.subheader("Avant / Apres")
                before_col, after_col = st.columns(2)
                before_col.image(before_preview, caption="Avant", use_container_width=True)
                after_col.image(after_preview, caption="Apres", use_container_width=True)
                export_preset = render_export_buttons()
                after_preview = adapt_to_social_canvas(processed_image, export_preset.image_size)
                image_bytes, output_name, mime = encode_image(after_preview, uploaded_file.name, export_preset)

                st.download_button(
                    "Telecharger la photo transformee",
                    data=image_bytes,
                    file_name=output_name,
                    mime=mime,
                    type="primary",
                )

            except Exception as exc:
                st.error(f"Le traitement de la photo a echoue : {exc}")


if __name__ == "__main__":
    main()
