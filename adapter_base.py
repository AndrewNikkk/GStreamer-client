from abc import ABC, abstractmethod
from collections import deque
import logging
import threading
import time
from typing import Optional
import gi
import numpy as np
import cv2

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GLib", "2.0")

from gi.repository import Gst, GLib, GstApp


class GStreamerAdapter(ABC):
    """Абстрактный класс адаптера GStreamer 
       с частичной реализацией
    """
    def __init__(self, 
                 source_path: str, 
                 width: int = 640, 
                 height: int = 480, 
                 fps: int = 30, 
                 user: Optional[str] = None, 
                 password: Optional[str] = None,
                 decoder: str = "cpu",
                 extra_props: dict = {}):
        
        self.source_path = source_path
        self.width = width
        self.height = height
        self.fps = fps
        self.user = user
        self.password = password
        self.decoder = decoder
        self.extra_props = extra_props
        
        self.frame_buffer = deque(maxlen=5)
        
        self.pipeline = None
        self.loop = None
        self.loop_thread = None
        self.converter = None
        self.appsink = None

        self.latest_frame = None
        self.last_frame_time = 0.0
        self.lock = threading.Lock()

        self.soft_timeout = 1.0
        self.hard_timeout = 2.0

        self.stub_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(self.stub_frame, "NO SIGNAL / CONNECTING...", (50, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def is_frame_fresh(self) -> bool:
        """
        Флаг-свойство: True, если кадр свежий и поток живой.
        False, если кадры задерживаются дольше, чем soft_timeout.
        """
        with self.lock:
            if self.latest_frame is None:
                return False
            return (time.time() - self.last_frame_time) <= self.soft_timeout


    @abstractmethod
    def initialize_pipeline(self):
        """
        Сборка пайплайна через анализ источника
        """
        pass


    def _build_appsink(self):
        """Создаёт appsink для получения кадров"""
        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        if not self.appsink:
            self.logger.error("Не удалось создать appsink")
            return
        
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("drop", True)
        self.appsink.set_property("max-buffers", 1)
        
        caps = Gst.Caps.from_string("video/x-raw, format=BGR")
        self.appsink.set_property("caps", caps)
        self.appsink.connect("new-sample", self._on_new_sample)
        
        self.pipeline.add(self.appsink)
        self.logger.info("Appsink создан и добавлен в пайплайн")


    def _on_new_sample(self, appsink: Gst.Element) -> Gst.FlowReturn:
        """
        Коллбэк, получает кадры из GStreamer и 
        конвертирует их в numpy массив без утечек памяти
        """
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

        numpy_frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3)).copy()

        buffer.unmap(map_info)
        
        with self.lock:
            self.latest_frame = numpy_frame
            self.last_frame_time = time.time()

        return Gst.FlowReturn.OK


    def start(self):
        """Запуск пайплайна"""
        with self.lock:
            self.latest_frame = None
            self.last_frame_time = 0.0


        if self.pipeline is None:
            self.initialize_pipeline()

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)

        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run)
        self.loop_thread.daemon = True
        self.loop_thread.start()
        self.logger.info("Пайплайн запущен")


    def restart(self):
        """Полный перезапуск пайплайна"""

        self.logger.warning("!!! Инициирован перезапуск паплайна GStreamer !!!")

        if self.loop:
            self.loop.quit()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        if self.loop_thread:
            self.loop_thread.join(timeout=1)

        self.pipeline = None
        self.src = None
        self.depay = None
        self.parser = None
        self.decoder_element = None
        self.converter = None
        self.appsink = None

        self.start()


    def stop(self):
        """Остановка пайплайна"""
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
                    self.logger.warning(f"Поток отвалился! Задержка: {time_passed:.2f} сек. Выводим заглушку")
                return self.stub_frame
            
            return self.latest_frame


    def _on_bus_message(self, bus, message):
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug_info = message.parse_error()
            self.logger.error(f"ОШИБКА от элемента {message.src.get_name()}: {err.message}")
            self.logger.error(f"Отладочная информация: {debug_info}")

        elif msg_type == Gst.MessageType.WARNING:
            err, debug_info = message.parse_warning()
            self.logger.warning(f"ПРЕДУПРЕЖДЕНИЕ от элемента {message.src.get_name()}: {err.message}")
            self.logger.warning(f"Отладочная информация: {debug_info}")

        elif msg_type == Gst.MessageType.EOS:
            self.logger.info("Поток данных завершен (End of Stream).")