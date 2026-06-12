import logging
import time
from typing import Optional
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from mod.adapter_base import GStreamerAdapter

Gst.init(None)

class RTSPadapter(GStreamerAdapter):
    """
    Адаптер GStreamer, принимающий RTSP поток
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.src = None
        self.depay = None
        self.parser = None
        self.decoder_element = None
        
        self.initialize_pipeline()


    def initialize_pipeline(self):
        self.pipeline = Gst.Pipeline.new('rtsp-pipeline')
        
        self._create_converter()
        self._build_appsink()

        if self.converter and self.appsink:
            self.logger.info("Связываем статические элементы: converter -> appsink")
            if not self.converter.link(self.appsink):
                self.logger.error("Не удалось связать converter с appsink!")
        
        self._build_source()


    def _build_source(self):
        """Создание и настройка элемента источника"""
        self.src = Gst.ElementFactory.make("rtspsrc", "source")
        if not self.src:
            self.logger.error("Не удалось создать элемент rtspsrc")
            return

        self.src.set_property("location", self.source_path)

        if self.user:
            self.src.set_property("user-id", self.user)

        if self.password:
            self.src.set_property("user-pw", self.password)
        
        if self.extra_props:
            for k, v in self.extra_props.items():
                if k.startswith("rtspsrc_"):
                    self.src.set_property(k[8:], v)
        
        self.pipeline.add(self.src)
        self.src.connect("pad-added", self._on_rtspsrc_pad_added, None)
        self.logger.info("Элемент rtspsrc добавлен и сконфигурирован")


    def _on_rtspsrc_pad_added(self, src_element, new_pad, user_data):
        self.logger.info(f"Появился новый динамический пад: {new_pad.get_name()}")

        if not self._is_video_pad(new_pad):
            return
        
        encoding = self._get_encoding_from_pad(new_pad)
        if not encoding:
            self.logger.warning("Не удалось определить кодек")
            return
        
        if not self._create_decoding_chain(encoding):
            self.logger.warning("Не удалось создать цепочку декодирования")
            return
        
        if not self._link_decoding_chain(new_pad):
            self.logger.warning("Не удалось связать элементы в цепочке декодирования")
            return
        
        self.logger.info(f"Динамическая RTSP цепочка успешно готова и запущена: {encoding}")


    def _is_video_pad(self, pad) -> bool:
        """Проверяет, является ли пад видеопотоком"""
        caps = pad.query_caps(None)
        if not caps or caps.is_empty():
            self.logger.error("Не удалось получить caps с пада")
            return False
        
        caps_struct = caps.get_structure(0)
        media_type = caps_struct.get_string("media")
        
        if media_type != "video":
            self.logger.info(f"Пропускаем не-видео пад: {media_type}")
            return False
        
        return True


    def _get_encoding_from_pad(self, pad) -> Optional[str]:
        """Определяет кодек с пада"""
        caps = pad.query_caps(None)
        caps_struct = caps.get_structure(0)
        encoding = caps_struct.get_string("encoding-name")
        
        if encoding not in ["H264", "H265"]:
            self.logger.error(f"Неподдерживаемый кодек: {encoding}")
            return None
        
        self.logger.info(f"Детектирован кодек: {encoding}")
        return encoding


    def _create_decoding_chain(self, encoding: str) -> bool:
        """
        Динамически создает элементы для декодирования и выставляет им 
        актуальное состояние родительского пайплайна
        """
        if not self._create_depay_and_parser(encoding):
            return False
        
        if not self._create_decoder(encoding):
            return False
        
        for elem in [self.depay, self.parser, self.decoder_element]:
            self.pipeline.add(elem)
            elem.sync_state_with_parent()
        
        return True


    def _create_depay_and_parser(self, encoding: str) -> bool:
        """Создает depay и parser элементы"""
        if encoding == "H264":
            depay_name = "rtph264depay"
            parser_name = "h264parse"
        elif encoding == "H265":
            depay_name = "rtph265depay"
            parser_name = "h265parse"
        else:
            return False
        
        self.depay = Gst.ElementFactory.make(depay_name, f"{encoding.lower()}-depay")
        self.parser = Gst.ElementFactory.make(parser_name, f"{encoding.lower()}-parser")
        
        if not self.depay or not self.parser:
            self.logger.error(f"Не удалось создать {depay_name} или {parser_name}")
            return False
        
        return True


    def _create_decoder(self, encoding: str) -> bool:
        """Создает декодер (CUDA или CPU)"""
        if encoding == "H264":
            decoder_cuda = "nvh264dec"
            decoder_cpu = "avdec_h264"
        else:
            decoder_cuda = "nvh265dec"
            decoder_cpu = "avdec_h265"
        
        if self.decoder == "cuda":
            decoder_name = decoder_cuda
            self.decoder_element = Gst.ElementFactory.make(decoder_name, "decoder")
            
            if not self.decoder_element:
                self.logger.warning(f"Не удалось создать CUDA декодер {decoder_name}, откатываемся на CPU")
                decoder_name = decoder_cpu
                self.decoder_element = Gst.ElementFactory.make(decoder_name, "decoder")
        else:
            decoder_name = decoder_cpu
            self.decoder_element = Gst.ElementFactory.make(decoder_name, "decoder")
        
        if not self.decoder_element:
            self.logger.error(f"Не удалось создать декодер: {decoder_name}")
            return False
        
        self.logger.info(f"Создан декодер: {decoder_name}")
        return True


    def _link_decoding_chain(self, new_pad) -> bool:
        """
        Связывает динамическую цепочку:
        new_pad (rtspsrc) -> depay -> parser -> decoder -> converter
        """
        if not self._link_pad_to_depay(new_pad):
            return False
        
        if not self.depay.link(self.parser):
            self.logger.error("Не удалось связать depay -> parser")
            return False
        
        if not self.parser.link(self.decoder_element):
            self.logger.error("Не удалось связать parser -> decoder")
            return False
        
        if self.converter and not self.decoder_element.link(self.converter):
            self.logger.error("Не удалось связать decoder -> converter")
            return False
        
        self.logger.info("Все динамические элементы успешно связаны с конвертером")
        return True


    def _link_pad_to_depay(self, new_pad) -> bool:
        """Связывает динамический пад rtspsrc с депайлоудером"""
        sink_pad = self.depay.get_static_pad("sink")
        
        if not sink_pad:
            self.logger.error("У depay нет sink пада")
            return False
        
        if sink_pad.is_linked():
            self.logger.warning("Sink пад depay уже с чем-то связан")
            return False
        
        ret = new_pad.link(sink_pad)
        if ret != Gst.PadLinkReturn.OK:
            self.logger.error(f"Не удалось связать rtspsrc пад с depay: {ret}")
            return False
        
        return True


    def _create_converter(self):
        """Создает видео-конвертер и сразу добавляет его в пайплайн"""
        if self.decoder == "cuda":
            converter = Gst.ElementFactory.make("nvconv", "gpu-converter")
            if not converter:
                self.logger.warning("Не удалось создать nvconv, переключаюсь на videoconvert")
                converter = Gst.ElementFactory.make("videoconvert", "cpu-converter")
        else:
            converter = Gst.ElementFactory.make("videoconvert", "cpu-converter")
        
        if not converter:
            self.logger.error("Не удалось создать элемент конвертера")
            return
        
        self.pipeline.add(converter)
        self.converter = converter


if __name__ == "__main__":
    import cv2
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    adapter = RTSPadapter(
        source_path="rtsp://10.153.240.130:8080/h264.sdp", 
        width=1920, 
        height=1080, 
        fps=30, 
        decoder="cpu",
        extra_props={"rtspsrc_protocols": 4}
    )

    adapter.start()

    try:
        while True:
            frame = adapter.get_image()

            if frame is not None:
                cv2.imshow("GStreamer RTSP Test", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.001)
    finally:
        adapter.stop()
        cv2.destroyAllWindows()