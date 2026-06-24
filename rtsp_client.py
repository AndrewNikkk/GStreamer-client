import logging
import time
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from gs_client_base import ElementSpec, GStreamerClient

Gst.init(None)


CODEC_CHAIN = {
    "H264": {
        "depay": "rtph264depay",
        "parser": "h264parse",
        "decoder_cuda": "nvh264dec",
        "decoder_cpu": "avdec_h264",
    },
    "H265": {
        "depay": "rtph265depay",
        "parser": "h265parse",
        "decoder_cuda": "nvh265dec",
        "decoder_cpu": "avdec_h265",
    },
}


class RTSPClient(GStreamerClient):
    """GStreamer-клиент для RTSP-потока."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.src = None
        self.depay = None
        self.parser = None
        self.decoder_element = None

        if not self.initialize_pipeline():
            raise RuntimeError("Не удалось собрать RTSP-пайплайн")


    def initialize_pipeline(self) -> bool:
        self.pipeline = Gst.Pipeline.new("rtsp-pipeline")

        if self.decoder == "cuda":
            if not self._build_gpu_converter():
                self.logger.warning("GPU-конвертер недоступен, откат на CPU")
                if not self._build_cpu_converter():
                    return False
        else:
            if not self._build_cpu_converter():
                return False

        if not self._build_appsink():
            return False

        if not self._link_chain(self.converter_out, self.appsink):
            return False

        return self._build_rtspsrc()


    def _build_cpu_converter(self) -> bool:
        specs = [
            ElementSpec("resize", "videoscale", "cpu-resize", {"method": 0}),
            ElementSpec("color", "videoconvert", "cpu-color"),
        ]

        if self.fps and self.fps > 0:
            specs.append(
                ElementSpec("rate", "videorate", "cpu-rate", {"drop-only": True})
            )

        specs.append(
            ElementSpec("caps", "capsfilter", "cpu-caps", {"caps": self._output_caps()})
        )

        src, out = self._build_static_chain(specs)

        if not src:
            return False
        
        self.converter_src = src
        self.converter_out = out

        self.logger.info(
            f"CPU-цепочка собрана: {self.width}x{self.height} BGR"
            + (f", {self.fps} fps" if self.fps and self.fps > 0 else "")
        )

        return True


    def _build_gpu_converter(self) -> bool:
        specs = [
            ElementSpec("resize", "nvconv", "gpu-resize"),
            ElementSpec(
                "caps",
                "capsfilter",
                "gpu-caps",
                {
                    "caps": Gst.Caps.from_string(
                        f"video/x-raw, width={self.width}, height={self.height}"
                    )
                },
            ),
            ElementSpec("color", "videoconvert", "gpu-color"),
        ]

        if self.fps and self.fps > 0:
            specs.append(
                ElementSpec("rate", "videorate", "gpu-rate", {"drop-only": True})
            )

        specs.append(
            ElementSpec("caps", "capsfilter", "gpu-out-caps", {"caps": self._output_caps()})
        )

        src, out = self._build_static_chain(specs)

        if not src:
            return False
        
        self.converter_src = src
        self.converter_out = out

        self.logger.info(
            f"GPU-цепочка собрана: {self.width}x{self.height} BGR"
            + (f", {self.fps} fps" if self.fps and self.fps > 0 else "")
        )

        return True


    def _build_rtspsrc(self) -> bool:
        self.src = self._make_element(
            "rtspsrc",
            "rtspsrc",
            "source",
            defaults={"location": self.source_path},
        )
        if not self.src:
            return False

        if self.user:
            self.src.set_property("user-id", self.user)
        if self.password:
            self.src.set_property("user-pw", self.password)

        self.pipeline.add(self.src)
        self.src.connect("pad-added", self._on_rtspsrc_pad_added, None)
        self.logger.info("rtspsrc добавлен и сконфигурирован")
        return True


    def _on_rtspsrc_pad_added(self, src_element, new_pad, user_data) -> None:
        self.logger.info(f"Появился динамический пад: {new_pad.get_name()}")

        if not self._is_video_pad(new_pad):
            return

        encoding = self._get_encoding_from_pad(new_pad)
        if not encoding:
            return

        if not self._create_decoding_chain(encoding):
            return

        if not self._link_decoding_chain(new_pad):
            return

        self.logger.info(f"Динамическая цепочка готова: {encoding}")


    def _is_video_pad(self, pad) -> bool:
        caps = pad.query_caps(None)
        if not caps or caps.is_empty():
            self.logger.error("Не удалось получить caps с пада")
            return False

        media_type = caps.get_structure(0).get_string("media")
        if media_type != "video":
            self.logger.info(f"Пропускаем не-видео пад: {media_type}")
            return False
        return True


    def _get_encoding_from_pad(self, pad) -> Optional[str]:
        encoding = pad.query_caps(None).get_structure(0).get_string("encoding-name")

        if encoding not in CODEC_CHAIN:
            self.logger.error(f"Неподдерживаемый кодек: {encoding}")
            return None

        self.logger.info(f"Детектирован кодек: {encoding}")
        return encoding

    def _create_decoding_chain(self, encoding: str) -> bool:
        chain = CODEC_CHAIN[encoding]

        self.depay = self._make_element(
            "depay",
            chain["depay"],
            f"{encoding.lower()}-depay",
        )
        self.parser = self._make_element(
            "parser",
            chain["parser"],
            f"{encoding.lower()}-parser",
        )

        fallbacks = [chain["decoder_cpu"]] if self.decoder == "cuda" else None
        preferred = (
            chain["decoder_cuda"] if self.decoder == "cuda" else chain["decoder_cpu"]
        )
        self.decoder_element = self._make_element(
            "decoder",
            preferred,
            "decoder",
            fallbacks=fallbacks,
        )

        if not all([self.depay, self.parser, self.decoder_element]):
            return False

        self._add_to_pipeline(self.depay, self.parser, self.decoder_element, sync=True)
        return True


    def _link_decoding_chain(self, new_pad) -> bool:
        if not self._link_pad_to_depay(new_pad):
            return False

        if not self._link_chain(self.depay, self.parser, self.decoder_element):
            return False

        if not self.converter_src:
            self.logger.error("converter_src не задан")
            return False

        if not self._link(
            self.decoder_element,
            self.converter_src,
            "decoder -> converter_src",
        ):
            return False

        self.logger.info("Динамическая цепочка связана с конвертером")
        return True


    def _link_pad_to_depay(self, new_pad) -> bool:
        sink_pad = self.depay.get_static_pad("sink")
        if not sink_pad:
            self.logger.error("У depay нет sink-пада")
            return False

        if sink_pad.is_linked():
            self.logger.warning("Sink-пад depay уже связан")
            return False

        ret = new_pad.link(sink_pad)
        if ret != Gst.PadLinkReturn.OK:
            self.logger.error(f"Не удалось связать rtspsrc -> depay: {ret}")
            return False

        return True


    def restart(self) -> bool:
        self.src = None
        self.depay = None
        self.parser = None
        self.decoder_element = None
        return super().restart()


if __name__ == "__main__":
    import cv2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    adapter = RTSPClient(
        source_path="rtsp://example",
        width=1440,
        height=1280,
        fps=1,
        decoder="cpu",
        element_props={
            "rtspsrc": {
                "protocols": 4,
                "latency": 0,
                "drop-on-latency": True,
                "buffer-mode": 0,
            },
            "depay": {
                "request-keyframe": True,
            },
        },
    )

    adapter.start()

    disconnect_start_time = None
    reconnect_cooldown = 5.0

    try:
        while True:
            frame = adapter.get_image()

            if adapter.is_frame_fresh:
                disconnect_start_time = None
                cv2.putText(
                    frame, "ONLINE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
            else:
                if disconnect_start_time is None:
                    disconnect_start_time = time.time()

                time_disconnected = time.time() - disconnect_start_time

                if time_disconnected > reconnect_cooldown:
                    print("Связь потеряна. Переподключаемся...")
                    adapter.restart()
                    disconnect_start_time = time.time()

                cv2.putText(
                    frame,
                    f"DISCONNECTED: {time_disconnected:.1f}s. RECONNECTING...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("GStreamer RTSP Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if not adapter.is_frame_fresh:
                time.sleep(0.1)

    finally:
        adapter.stop()
        cv2.destroyAllWindows()