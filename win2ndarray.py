import struct
import numpy as np
import pandas as pd
import mmap
from collections import defaultdict
from datetime import datetime

def bcd_to_int(b):
    return ((b >> 4) * 10) + (b & 0x0F)

def _decode_diff_vectorized(sample_size_code: int, num_samples: int, initial_sample: int, raw_data: memoryview | bytes) -> np.ndarray:
    """差分列をベクトル化で復号し、累積和で元波形に戻す（int32返し）"""
    remaining = num_samples - 1
    if remaining <= 0:
        return np.array([initial_sample], dtype=np.int32)

    if sample_size_code == 0:
        # 4bit 符号付き (-8..+7) を 2 サンプル/byte で詰め込み
        b = np.frombuffer(raw_data, dtype=np.uint8)
        # 上位/下位ニブルに展開
        hi = (b >> 4) & 0x0F
        lo = b & 0x0F
        nibbles = np.empty(b.size * 2, dtype=np.int8)
        nibbles[0::2] = hi
        nibbles[1::2] = lo
        # 符号拡張（>=8 は負）
        neg = nibbles >= 8
        nibbles = nibbles.astype(np.int8, copy=False)
        nibbles[neg] -= 16
        diffs = nibbles[:remaining].astype(np.int32, copy=False)

    elif sample_size_code == 1:
        diffs = np.frombuffer(raw_data, dtype=np.int8, count=remaining).astype(np.int32, copy=False)

    elif sample_size_code == 2:
        diffs = np.frombuffer(raw_data, dtype=">i2", count=remaining).astype(np.int32, copy=False)

    elif sample_size_code == 3:
        # 24bit signed: 3 バイトをまとめて 0x800000 で判定し 1<<24 を引く
        u = np.frombuffer(raw_data, dtype=np.uint8, count=remaining*3)
        u = u.reshape(-1, 3)
        vals = (u[:,0].astype(np.uint32) << 16) | (u[:,1].astype(np.uint32) << 8) | u[:,2].astype(np.uint32)
        vals = vals.astype(np.int32, copy=False)
        mask = (vals & 0x800000) != 0
        vals[mask] -= (1 << 24)
        diffs = vals

    elif sample_size_code == 4:
        diffs = np.frombuffer(raw_data, dtype=">i4", count=remaining).astype(np.int32, copy=False)

    else:
        # 未知コードは空
        diffs = np.empty(0, dtype=np.int32)

    # 累積和＋初期値
    out = np.empty(num_samples, dtype=np.int32)
    out[0] = initial_sample
    if diffs.size:
        np.cumsum(diffs, dtype=np.int64, out=diffs)   # オーバーフロー安全に
        out[1:] = initial_sample + diffs
    return out

def parse_win_file(file_path):
    """
    WIN ファイルを高速にパースして
      channel_data: {ch_id: np.int32 波形（連結済み）}
      time_index  : {ch_id: np.datetime64[us] の時刻列（連結済み）}
    を返す。
    ※ memoryview を使わず、np.frombuffer には bytes を渡すことで BufferError を回避。
    """
    channel_arrays = defaultdict(list)
    time_arrays = defaultdict(list)

    with open(file_path, "rb") as f:
        buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        offset = 0
        end = len(buf)

        while offset + 10 <= end:
            # 先頭4バイト: ブロック長
            block_len = struct.unpack_from(">I", buf, offset)[0]
            # 次の6バイト: BCD 時刻（bytes を取得）
            ts_bytes = buf[offset+4:offset+10]  # これは bytes（mmap のスライスは bytes）
            year  = bcd_to_int(ts_bytes[0]) + 2000
            month = bcd_to_int(ts_bytes[1])
            day   = bcd_to_int(ts_bytes[2])
            hour  = bcd_to_int(ts_bytes[3])
            minute= bcd_to_int(ts_bytes[4])
            second= bcd_to_int(ts_bytes[5])

            base_time = np.datetime64(
                f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}", "us"
            )

            offset += 10
            block_end = offset + (block_len - 10)
            if block_end > end:
                break

            ptr = offset
            while ptr + 4 <= block_end:
                ch_id = struct.unpack_from(">H", buf, ptr)[0]
                size_rate = struct.unpack_from(">H", buf, ptr+2)[0]
                sample_size_code = (size_rate & 0xF000) >> 12
                sampling_rate = size_rate & 0x0FFF
                ptr += 4

                if sampling_rate == 0 or ptr + 4 > block_end:
                    break

                num_samples = sampling_rate
                initial_sample = struct.unpack_from(">i", buf, ptr)[0]
                ptr += 4

                data_len = {
                    0: (num_samples - 1 + 1) // 2,
                    1: (num_samples - 1),
                    2: (num_samples - 1) * 2,
                    3: (num_samples - 1) * 3,
                    4: (num_samples - 1) * 4
                }.get(sample_size_code, 0)

                if ptr + data_len > block_end:
                    break

                # ★ np.frombuffer に渡すのは bytes（コピー）にする
                raw_bytes = bytes(buf[ptr:ptr+data_len])
                ptr += data_len

                decoded = _decode_diff_vectorized(
                    sample_size_code, num_samples, initial_sample, raw_bytes
                )

                step_us = int(round(1_000_000 / sampling_rate))
                t = base_time + (np.arange(num_samples, dtype=np.int64) * step_us).astype("timedelta64[us]")

                channel_arrays[ch_id].append(decoded)
                time_arrays[ch_id].append(t)

            offset = block_end

        # ★ memoryview を使っていないので、そのまま閉じて OK
        buf.close()

    channel_data = {k: np.concatenate(v).astype(np.int32, copy=False) for k, v in channel_arrays.items()}
    time_index   = {k: np.concatenate(v).astype("datetime64[us]", copy=False) for k, v in time_arrays.items()}
    return channel_data, time_index

def load_ch_table(fn):
    output_list = []
    with open(fn, "r") as f:
        for line in f:
            output = []
            if not line.isspace():
                if '#' not in line.split()[0]:
                    try:
                        output = [line.split()[0], line.split()[1], line.split()[2], line.split()[3],
                                  line.split()[4], line.split()[5], line.split()[6], float(line.split()[7]),
                                  line.split()[8], line.split()[9], line.split()[10], float(line.split()[11]),
                                  float(line.split()[12])]

                        if len(line.split()) > 14:
                            if '-' in line.split()[14]:
                                output.append(float(line.split()[13]))
                                output.append(float(line.split()[14].split('-')[0]))
                                output.append(float('-' + line.split()[14].split('-')[1]))
                            else:
                                output.append(float(line.split()[13]))
                                output.append(float(line.split()[14]))
                                output.append(float(line.split()[15]))
                        else:
                            output.append(0)
                            output.append(0)
                            output.append(0)

                        output_list.append(output)
                    except:
                        continue

    df = pd.DataFrame(output_list, columns=['ChID', 'flag', 'delay', 'stname', 'comp', 'monitor_amp', 'bit',
                                            'sensitivity', 'unit', 'nat_period', 'damping', 'gain_dB', 'step_width',
                                            'lat', 'lon', 'elv'])
    return df

def extract_waveform_metadata(win_file_path, ch_file_path):
    parsed_channel_data, time_index = parse_win_file(win_file_path)
    ch_df = load_ch_table(ch_file_path)

    waveform_records = []
    waveform_dict = defaultdict(dict)

    for _, row in ch_df.iterrows():
        logical_id = int(row["ChID"], 16)
        station = row['stname']
        component = row['comp']

        step_with = row['step_width']
        sensitivity = row['sensitivity']
        gain_dB = row['gain_dB']

        raw_waveform = np.array(parsed_channel_data.get(logical_id, [0]), dtype=np.float32)

        if len(raw_waveform) == 1:
            continue

        physical_waveform = (raw_waveform * step_with) / (sensitivity * 10 ** (gain_dB / 20))

        times = np.array(time_index.get(logical_id, [None] * len(raw_waveform)))
        if times[0] is None:
            continue

        time_step = (times - times[0]) / np.timedelta64(1, 's')
        
        if component == 'X':
            component = 'E'
            
        if component == 'Y':
            component = 'N'

        waveform_dict[station][component] = physical_waveform
        waveform_dict[station]['time'] = times
        waveform_dict[station]['time_step'] = time_step

        waveform_records.append({
            "Station": station,
            "Component": component,
            "LogicalCh": logical_id,
            "WaveformLength": len(raw_waveform),
            "Unit": row["unit"],
            "Lat": row["lat"],
            "Lon": row["lon"],
            "Elv":row["elv"]
        })

    waveform_info_df = pd.DataFrame(waveform_records)
    return waveform_info_df, waveform_dict
