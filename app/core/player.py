"""播放引擎：sounddevice 多通道输出 + 环形缓冲预取 + 位置/跳转/循环。

实时回调只做「环形缓冲 → outdata」的拷贝；磁盘读取（可能是 memmap 缺页）放在独立
预取线程里，避免卡住 PortAudio 回调造成爆音。

位置语义
--------
``_pos`` 表示「已送去 DAC 的帧数」，是播放指针的唯一真相；预取指针 ``_fill_pos`` 总是
领先于它。循环时 ``_pos`` 在越过 ``b`` 的瞬间折回 ``a``，与环形缓冲里被预取线程重写的
数据顺序天然一致。
"""

from __future__ import annotations

import threading
import time
from typing import Sequence

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - 无 PortAudio 时只禁用播放
    sd = None


def list_output_devices() -> list[dict]:
    if sd is None:
        return []
    try:
        default_out = sd.default.device[1] if sd.default.device else -1
    except Exception:
        default_out = -1
    out = []
    for i, d in enumerate(sd.query_devices()):
        if not d.get("max_output_channels", 0):
            continue
        try:
            rate = int(round(d.get("default_samplerate", 0) or 0))
        except Exception:
            rate = 0
        out.append({
            "index": i,
            "name": d.get("name", f"device {i}"),
            "channels": int(d.get("max_output_channels", 0)),
            "rate": rate,
            "is_default": i == default_out,
        })
    return out


class Player:
    """多通道播放器。UI 通过 ``play/pause/seek/position`` 与之交互。"""

    RING_FRAMES = 32768

    def __init__(self, project) -> None:
        self.project = project
        self.rate = 48000
        self.lanes: list = []
        self.channels = 0
        self.total_frames = 0
        self.device: int | None = None
        self.blocksize = 2048
        self.volume = 1.0
        self.muted = False
        self.loop: tuple[int, int] | None = None
        self.error: str | None = None
        self.underruns = 0
        # 实时诊断计数
        self.cb_count = 0
        self.cb_frames = 0
        self.cb_empty = 0
        self.cb_t0: float | None = None
        self.cb_t_last: float | None = None
        self.cb_blocks: set[int] = set()

        self._lock = threading.Lock()
        self._ring = np.zeros((self.RING_FRAMES, 1), dtype=np.float32)
        self._wpos = 0
        self._rpos = 0
        self._avail = 0
        self._pos = 0
        self._fill_pos = 0
        self._playing = False
        self._stop_evt = threading.Event()
        self._generation = 0
        self._stream = None
        self._filler: threading.Thread | None = None
        self._eof = False

    # ------------------------------------------------------------------
    # 准备
    # ------------------------------------------------------------------
    def prepare(self, lanes: Sequence, rate: int, device: int | None = None) -> tuple[int, str | None]:
        """绑定播放通道集合，返回实际通道数与告警。"""
        was_playing = self.is_playing()
        if was_playing:
            self.stop()
        self.rate = int(rate) or 48000
        self.lanes = list(lanes)
        self.device = device
        warn = None
        if sd is not None and device is not None:
            try:
                have = int(sd.query_devices(device)["max_output_channels"])
            except Exception:
                have = len(self.lanes)
            if len(self.lanes) > have >= 1:
                warn = f"输出设备仅 {have} 个通道，按前 {have}/{len(self.lanes)} 个通道播放"
                self.lanes = self.lanes[:have]
        self.channels = max(1, len(self.lanes))
        self.total_frames = int(self.project.total_frames)
        self.loop = None
        self._eof = False
        with self._lock:
            self._ring = np.zeros((self.RING_FRAMES, self.channels), dtype=np.float32)
            self._clear_ring_locked(self._pos if self._pos < self.total_frames else 0)
        return self.channels, warn

    def _clear_ring_locked(self, pos: int) -> None:
        self._ring[:] = 0
        self._wpos = 0
        self._rpos = 0
        self._avail = 0
        self._pos = max(0, min(int(pos), max(0, self.total_frames)))
        self._fill_pos = self._pos
        self._generation += 1

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    def is_playing(self) -> bool:
        return bool(self._playing)

    @property
    def position(self) -> int:
        return int(self._pos)

    @property
    def position_seconds(self) -> float:
        return self._pos / self.rate if self.rate else 0.0

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.rate if self.rate else 0.0

    # ------------------------------------------------------------------
    # 传输
    # ------------------------------------------------------------------
    def play(self, at_frames: int | None = None) -> bool:
        if sd is None:
            self.error = "未安装 sounddevice / PortAudio，无法播放"
            return False
        if self.total_frames <= 0 or not self.lanes:
            self.error = "没有可播放的通道：请先打开文件并勾选至少一个通道"
            return False
        self.error = None
        if at_frames is not None:
            self._do_seek(at_frames)
        elif self._pos >= self.total_frames - 1 or self._eof:
            self._do_seek(self.loop[0] if self.loop else 0)
            self._eof = False
        if self._stream is None:
            try:
                self._stream = sd.OutputStream(
                    samplerate=self.rate,
                    blocksize=self.blocksize,
                    channels=self.channels,
                    dtype="float32",
                    device=self.device,
                    callback=self._callback,
                )
            except Exception as exc:
                self.error = f"打开音频输出失败：{exc}"
                self._stream = None
                return False
        self._stop_evt.clear()
        self._playing = True
        self._start_filler()
        try:
            self._stream.start()
        except Exception as exc:
            self.error = f"启动音频输出失败：{exc}"
            self._playing = False
            return False
        return True

    def pause(self) -> None:
        self._playing = False
        self._stop_evt.set()
        thread, self._filler = self._filler, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        try:
            if self._stream is not None:
                self._stream.stop()
        except Exception:
            pass

    def stop(self) -> None:
        self.pause()
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        self._eof = False
        with self._lock:
            self._clear_ring_locked(0)

    def toggle(self) -> bool:
        if self.is_playing():
            self.pause()
            return False
        return self.play()

    def seek(self, frames: int) -> None:
        self._do_seek(frames)

    def seek_seconds(self, seconds: float) -> None:
        self._do_seek(int(round(seconds * self.rate)))

    def _do_seek(self, frames: int) -> None:
        frames = max(0, min(int(frames), max(0, self.total_frames - 1))) if self.total_frames else 0
        with self._lock:
            self._eof = False
            self._clear_ring_locked(frames)
        if self._playing:
            self._start_filler()

    def _start_filler(self) -> None:
        thread, self._filler = self._filler, None
        if thread is not None and thread.is_alive():
            self._stop_evt.clear()          # 旧线程靠 generation 自行退出
        gen = self._generation
        self._stop_evt.clear()
        self._filler = threading.Thread(target=self._fill_loop, args=(gen,),
                                        name="player-prefetch", daemon=True)
        self._filler.start()

    def set_volume(self, v: float) -> None:
        self.volume = float(max(0.0, min(2.0, v)))

    def set_loop(self, region: tuple[int, int] | None) -> None:
        if region is None:
            self.loop = None
            return
        a, b = int(min(region)), int(max(region))
        self.loop = (max(0, a), min(b, self.total_frames))
        if self.loop[1] - self.loop[0] < 2:
            self.loop = None

    # ------------------------------------------------------------------
    # 实时回调
    # ------------------------------------------------------------------
    def _callback(self, outdata, frames, time_info, status):  # pragma: no cover - 实时线程
        if status:
            self.underruns += 1
        gain = 0.0 if self.muted else self.volume
        with self._lock:
            avail = self._avail
            if avail <= 0:
                outdata[:] = 0
                self.cb_empty += 1
                return
            take = min(frames, avail)
            r = self._rpos
            span = min(take, self.ring_frames_local - r)
            if span >= take:
                block = self._ring[r:r + span]
            else:
                block = np.concatenate([self._ring[r:r + span], self._ring[: take - span]])
            outdata[:take] = block
            if take < frames:
                outdata[take:] = 0
            self._rpos = (r + take) % self.ring_frames_local
            self._avail -= take
            pos = self._pos + take
            if self.loop:
                a, b = self.loop
                length = max(1, b - a)
                if pos >= b:
                    pos = a + (pos - b) % length
            elif pos >= self.total_frames:
                pos = self.total_frames
            self._pos = pos
            now = time.monotonic()
            if self.cb_t0 is None:
                self.cb_t0 = now
            self.cb_t_last = now
            self.cb_count += 1
            self.cb_frames += take
            self.cb_blocks.add(take)
        if gain != 1.0:
            outdata[:take] *= gain

    def realtime_stats(self) -> dict:
        """实时诊断：回调次数、送出的帧数、墙上时长、等效帧率、空回调数。"""
        wall = (self.cb_t_last - self.cb_t0) if (self.cb_t0 and self.cb_t_last) else 0.0
        return {
            "callbacks": self.cb_count,
            "frames": self.cb_frames,
            "wall": wall,
            "fps": (self.cb_frames / wall) if wall > 0.05 else 0.0,
            "rate": self.rate,
            "blocksizes": sorted(self.cb_blocks),
            "empty": self.cb_empty,
            "underruns": self.underruns,
        }

    @property
    def ring_frames_local(self) -> int:
        return self._ring.shape[0]

    # ------------------------------------------------------------------
    # 预取线程
    # ------------------------------------------------------------------
    def _fill_loop(self, gen: int) -> None:
        ring = self.RING_FRAMES
        while self._playing and not self._stop_evt.is_set():
            with self._lock:
                if gen != self._generation:
                    return
                fill_pos = self._fill_pos
                free = ring - self._avail
            if self.total_frames and fill_pos >= self.total_frames and not self.loop:
                self._wait_drain(gen)
                return
            if self.loop and fill_pos >= self.loop[1]:
                with self._lock:
                    if gen != self._generation:
                        return
                    self._fill_pos = fill_pos = self.loop[0]
            limit = self.total_frames - fill_pos if self.total_frames else ring
            if self.loop:
                limit = min(limit, self.loop[1] - fill_pos)
            n = int(max(1, min(free, self.blocksize * 4, limit)))
            if free < self.blocksize:
                time.sleep(0.005)
                continue
            try:
                data = self.project.read(self.lanes, fill_pos, n, self.rate)
            except Exception as exc:  # pragma: no cover
                self.error = f"读取音频失败：{exc}"
                self._playing = False
                return
            with self._lock:
                if gen != self._generation:
                    return
                if not self._playing:
                    return
                w = self._wpos
                take = min(n, data.shape[0], ring - w)
                self._ring[w:w + take] = data[:take]
                if take < min(n, data.shape[0]):
                    rest = min(n, data.shape[0]) - take
                    self._ring[:rest] = data[take:take + rest]
                    take = min(n, data.shape[0])
                self._wpos = (w + take) % ring
                self._avail += take
                self._fill_pos = fill_pos + take

    def _wait_drain(self, gen: int) -> None:
        """文件读完：等缓冲放完再自然停在末尾。"""
        deadline = time.time() + 5.0
        while self._playing and not self._stop_evt.is_set() and time.time() < deadline:
            with self._lock:
                if gen != self._generation:
                    return
                if self._avail <= 0:
                    break
            time.sleep(0.02)
        self._playing = False
        self._eof = True
        try:
            if self._stream is not None:
                self._stream.stop()
        except Exception:
            pass
        with self._lock:
            self._pos = self.total_frames

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
