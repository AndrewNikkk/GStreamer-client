from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Optional
import gi
import numpy as np
import cv2

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import Gst, GLib, GstApp


@dataclass
class ElementSpec:
    """Описание одного элемента в статической цепочке."""
    slot: str
    factory: str
    name: str
    props: dict[str, Any] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)


class GStreamerClient(ABC):
    """Базовый клиент GStreamer: конфигурация, сборка элементов, runtime."""

    def __init__(
        self,
        source_path: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        user: Optional[str] = None,
        password: Optional[str] = None,
        decoder: str = "cpu",
        element_props: Optional[dict[str, dict[str, Any]]] = None,
        element_overrides: Optional[dict[str, str]] = None,
    ):
        self.source_path = source_path
        self.width = width
        self.height = height
        self.fps = fps
        self.user = user
        self.password = password
        self.decoder = decoder
        self.element_props = element_props or {}
        self.element_overrides = element_overrides or {}

        self.frame_buffer = deque(maxlen=5)

        self.pipeline: Optional[Gst.Pipeline] = None
        self.loop = None
        self.loop_thread = None
        self.appsink = None

        self.converter_src = None
        self.converter_out = None

        self.latest_frame = None
        self.last_frame_time = 0.0
        self.lock = threading.Lock()

        self.soft_timeout = 1.0
        self.hard_timeout = 2.0

        self.stub_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            self.stub_frame,
            "NO SIGNAL / CONNECTING...",
            (50, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------ #
    #  Движок сборки пайплайна                                           #
    # ------------------------------------------------------------------ #

    def _make_element(
        self,
        slot: str,
        factory: str,
        name: str,
        *,
        defaults: Optional[dict[str, Any]] = None,
        fallbacks: Optional[list[str]] = None,
    ) -> Optional[Gst.Element]:
        factory = self.element_overrides.get(slot, factory)

        elem = Gst.ElementFactory.make(factory, name)
        if not elem and fallbacks:
            for fallback_factory in fallbacks:
                self.logger.warning(
                    f"[{slot}] {factory} недоступен, пробуем {fallback_factory}"
                )
                elem = Gst.ElementFactory.make(fallback_factory, name)
                if elem:
                    factory = fallback_factory
                    break

        if not elem:
            self.logger.error(f"[{slot}] Не удалось создать элемент {factory}")
            return None

        for key, value in (defaults or {}).items():
            elem.set_property(key, value)

        for key, value in self.element_props.get(slot, {}).items():
            elem.set_property(key, value)

        self.logger.info(f"[{slot}] Создан {factory} ({name})")
        return elem

    def _link(self, src: Gst.Element, dst: Gst.Element, label: str) -> bool:
        if not src.link(dst):
            self.logger.error(f"Не удалось связать {label}")
            return False
        return True

    def _link_chain(self, *elements: Gst.Element) -> bool:
        for src, dst in zip(elements, elements[1:]):
            label = f"{src.get_name()} -> {dst.get_name()}"
            if not self._link(src, dst, label):
                return False
        return True

    def _add_to_pipeline(self, *elements: Gst.Element, sync: bool = False) -> None:
        for elem in elements:
            self.pipeline.add(elem)
            if sync:
                elem.sync_state_with_parent()

    def _build_static_chain(
        self, specs: list[ElementSpec]
    ) -> tuple[Optional[Gst.Element], Optional[Gst.Element]]:
        elements: list[Gst.Element] = []

        for spec in specs:
            elem = self._make_element(
                spec.slot,
                spec.factory,
                spec.name,
                defaults=spec.props,
                fallbacks=spec.fallbacks or None,
            )
            if not elem:
                return None, None
            elements.append(elem)

        self._add_to_pipeline(*elements)

        if not self._link_chain(*elements):
            return None, None

        return elements[0], elements[-1]

    def _output_caps(self) -> Gst.Caps:
        return Gst.Caps.from_string(
            f"video/x-raw, format=BGR, width={self.width}, height={self.height}"
        )

    def _build_appsink(self) -> bool:
        self.appsink = self._make_element(
            "appsink",
            "appsink",
            "appsink",
            defaults={
                "emit-signals": True,
                "sync": False,
                "drop": True,
                "max-buffers": 1,
                "caps": self._output_caps(),
            },
        )
        if not self.appsink:
            return False

        self.appsink.connect("new-sample", self._on_new_sample)
        self.pipeline.add(self.appsink)
        self.logger.info(
            f"Appsink добавлен в пайплайн. Целевое разрешение: {self.width}x{self.height}"
        )
        return True

    # ------------------------------------------------------------------ #
    #  Runtime                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_frame_fresh(self) -> bool:
        with self.lock:
            if self.latest_frame is None:
                return False
            return (time.time() - self.last_frame_time) <= self.soft_timeout

    @abstractmethod
    def initialize_pipeline(self) -> bool:
        """Сборка пайплайна. Возвращает True при успехе."""
        ...

    def _on_new_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        sample = appsink.pull_sample()
        if not sample:
            return Gst.FlowReturn.ERROR

        caps = sample.get_caps()
        caps_data = caps.get_structure(0)
        width = caps_data.get_value("width")
        height = caps_data.get_value("height")

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        numpy_frame = (
            np.frombuffer(map_info.data, dtype=np.uint8)
            .reshape((height, width, 3))
            .copy()
        )
        buffer.unmap(map_info)

        with self.lock:
            self.latest_frame = numpy_frame
            self.last_frame_time = time.time()

        return Gst.FlowReturn.OK

    def start(self) -> bool:
        with self.lock:
            self.latest_frame = None
            self.last_frame_time = 0.0

        if self.pipeline is None:
            if not self.initialize_pipeline():
                self.logger.error("Не удалось инициализировать пайплайн")
                return False

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)
        self.loop_thread.start()
        self.logger.info("Пайплайн запущен")
        return True

    def restart(self) -> bool:
        self.logger.warning("!!! Инициирован перезапуск пайплайна GStreamer !!!")

        if self.loop:
            self.loop.quit()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop_thread:
            self.loop_thread.join(timeout=1)

        self.pipeline = None
        self.appsink = None
        self.converter_src = None
        self.converter_out = None

        return self.start()

    def stop(self) -> None:
        self.logger.info("Остановка пайплайна...")
        if self.loop:
            self.loop.quit()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop_thread:
            self.loop_thread.join(timeout=2)
        self.logger.info("Пайплайн остановлен")

    def get_image(self):
        with self.lock:
            if self.latest_frame is None:
                return self.stub_frame

            time_passed = time.time() - self.last_frame_time
            if time_passed > self.soft_timeout:
                if int(time_passed) % 2 == 0:
                    self.logger.warning(
                        f"Поток отвалился! Задержка: {time_passed:.2f} сек. Выводим заглушку"
                    )
                return self.stub_frame

            return self.latest_frame

    def _on_bus_message(self, bus, message) -> None:
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug_info = message.parse_error()
            self.logger.error(
                f"ОШИБКА от элемента {message.src.get_name()}: {err.message}"
            )
            self.logger.error(f"Отладочная информация: {debug_info}")

        elif msg_type == Gst.MessageType.WARNING:
            err, debug_info = message.parse_warning()
            self.logger.warning(
                f"ПРЕДУПРЕЖДЕНИЕ от элемента {message.src.get_name()}: {err.message}"
            )
            self.logger.warning(f"Отладочная информация: {debug_info}")

        elif msg_type == Gst.MessageType.EOS:
            self.logger.info("Поток данных завершен (End of Stream).")