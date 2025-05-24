import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
# import ace_tools as tools
from numba import njit
from datetime import datetime, timedelta

def bcd_to_int(bcd_byte):
    return ((bcd_byte >> 4) * 10) + (bcd_byte & 0x0F)

@njit
def decode_diff_data(sample_size_code, num_samples, initial_sample, raw_data):
    data = [initial_sample]
    ptr = 0
    remaining = num_samples - 1

    if sample_size_code == 0:
        for _ in range((remaining + 1) // 2):
            byte = raw_data[ptr]
            ptr += 1
            for shift in (4, 0):
                diff = (byte >> shift) & 0x0F
                if diff & 0x08:
                    diff -= 0x10
                if len(data) < num_samples:
                    data.append(data[-1] + diff)
    elif sample_size_code == 1:
        for _ in range(remaining):
            diff = np.int8(raw_data[ptr])
            ptr += 1
            data.append(data[-1] + diff)
    elif sample_size_code == 2:
        for _ in range(remaining):
            diff = np.int16((raw_data[ptr] << 8) | raw_data[ptr + 1])
            ptr += 2
            data.append(data[-1] + diff)
    elif sample_size_code == 3:
        for _ in range(remaining):
            val = (raw_data[ptr] << 16) | (raw_data[ptr + 1] << 8) | raw_data[ptr + 2]
            if val & 0x800000:
                val -= 0x1000000
            ptr += 3
            data.append(data[-1] + val)
    elif sample_size_code == 4:
        for _ in range(remaining):
            diff = np.int32((raw_data[ptr] << 24) | (raw_data[ptr + 1] << 16) | (raw_data[ptr + 2] << 8) | raw_data[ptr + 3])
            ptr += 4
            data.append(data[-1] + diff)

    return data

def parse_win_file(file_path):
    channel_data = defaultdict(list)
    time_index = defaultdict(list)

    read = memoryview(open(file_path, 'rb').read())
    offset = 0
    end = len(read)

    while offset + 10 <= end:
        block_len = struct.unpack_from('>I', read, offset)[0]
        timestamp_bytes = read[offset + 4:offset + 10]
        year = bcd_to_int(timestamp_bytes[0]) + 2000
        month = bcd_to_int(timestamp_bytes[1])
        day = bcd_to_int(timestamp_bytes[2])
        hour = bcd_to_int(timestamp_bytes[3])
        minute = bcd_to_int(timestamp_bytes[4])
        second = bcd_to_int(timestamp_bytes[5])
        base_time = datetime(year, month, day, hour, minute, second)

        offset += 10
        block_end = offset + (block_len - 10)
        if block_end > end:
            break

        ptr = offset
        while ptr + 4 <= block_end:
            ch_id = struct.unpack_from('>H', read, ptr)[0]
            size_rate = struct.unpack_from('>H', read, ptr + 2)[0]
            sample_size_code = (size_rate & 0xF000) >> 12
            sampling_rate = size_rate & 0x0FFF
            ptr += 4

            if sampling_rate == 0 or ptr + 4 > block_end:
                break

            num_samples = sampling_rate
            initial_sample = struct.unpack_from('>i', read, ptr)[0]
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
            raw_data = np.frombuffer(read[ptr:ptr + data_len], dtype=np.uint8)
            ptr += data_len

            decoded = decode_diff_data(sample_size_code, num_samples, initial_sample, raw_data)
            channel_data[ch_id].extend(decoded)

            for i in range(num_samples):
                time_index[ch_id].append(base_time + timedelta(seconds=i / sampling_rate))

        offset = block_end

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

                        if line.split()[4] in ['U', 'wU', 'UA', 'VL', 'V']:
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
        
        physical_waveform = (raw_waveform * step_with)/(sensitivity*10**(gain_dB/20))
        
        times = time_index.get(logical_id, [None] * len(raw_waveform))
        time_step = [(t - times[0]).total_seconds() if t is not None else None for t in times]

        waveform_dict[station][component] = physical_waveform
        waveform_dict[station]['time'] = times
        waveform_dict[station]['time_step'] = time_step

        waveform_records.append({
            "Station": station,
            "Component": component,
            "LogicalCh": logical_id,
            "WaveformLength": len(raw_waveform),
            "Unit": row["unit"],
            "Latitude": row["lat"],
            "Longitude": row["lon"]
        })

    waveform_info_df = pd.DataFrame(waveform_records)
    return waveform_info_df, waveform_dict

if __name__ == '__main__':
    waveform_info_df, waveform_dict = extract_waveform_metadata("./180805.090417", "./180805.090417.ch")